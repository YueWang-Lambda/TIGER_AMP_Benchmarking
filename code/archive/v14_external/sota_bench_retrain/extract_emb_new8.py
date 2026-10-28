#!/usr/bin/env python3
"""One-shot frozen-PLM embedding extraction for the 8 new benchmark methods.

Universe: emb_cache/universe.fa (13,086 seqs = restricted4929 + 7,962 negatives
+ 208 external peptides; frozen backbones are deterministic -> extract once,
reuse across all 10 seeds).

Models:
  esm2_3b  esm2_t36_3B_UR50D, layer36 mean (2560d)      -> AMPGen   (npz)
  esm1v    esm1v_t33_650M_UR90S_1, layer33 mean (1280d) -> tsAMP-CS (npz)
  esm1b    esm1b_t33_650M_UR50S, layer33 per-token + contact map
           (per-seq .pt, repo format)                    -> AMPredictor
  prott5   Rostlab/prot_t5_xl_uniref50 encoder, per-residue 1024d
           (npz of packed variable-length arrays)        -> GGN-ABPMIC

Usage: python extract_emb_new8.py --model {esm2_3b,esm1v,esm1b,prott5} \
         [--toks_per_batch N] [--out NAME]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'emb_cache'


def load_fasta(p):
    seqs = {}
    name = None
    for line in open(p):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
        elif name:
            seqs[name] = line
    return seqs


def esm_extract(model_name, repr_layer, toks, per_token=False):
    import esm
    seqs = load_fasta(CACHE / 'universe.fa')
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()
    data = sorted(seqs.items(), key=lambda kv: len(kv[1]))
    out_mean, out_tok = {}, {}
    i, B = 0, 64
    n_done = 0
    with torch.no_grad():
        while i < len(data):
            batch = []
            used = 0
            while i < len(data) and (len(batch) < 512 and used < toks):
                L = len(data[i][1]) + 2
                if batch and used + L > toks:
                    break
                batch.append(data[i]); used += L; i += 1
            _, _, toks_t = bc(batch)
            toks_t = toks_t.cuda()
            res = model(toks_t, repr_layers=[repr_layer],
                        return_contacts=per_token)
            rep = res['representations'][repr_layer].float()
            for j, (nm, seq) in enumerate(batch):
                r = rep[j, 1:len(seq) + 1]
                if per_token:
                    out_tok[seq] = {'feature': r.cpu().clone(),
                                    'contacts': res['contacts'][j, :len(seq), :len(seq)].cpu().clone()}
                else:
                    out_mean[seq] = r.mean(0).cpu().numpy().astype(np.float32)
            n_done += len(batch)
            if n_done % 2048 < len(batch):
                print(f'[{model_name}] {n_done}/{len(data)}', flush=True)
    del model; torch.cuda.empty_cache()
    return out_mean, out_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['esm2_3b', 'esm1v', 'esm1b', 'prott5'])
    ap.add_argument('--toks_per_batch', type=int, default=4096)
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    if args.model == 'prott5':
        from transformers import T5Tokenizer, T5EncoderModel
        d = '/data4T/ubuntu/wangyue/models/Prot-T5-XL-U50'
        tok = T5Tokenizer.from_pretrained(d, do_lower_case=False)
        enc = T5EncoderModel.from_pretrained(d).eval().cuda().half()
        seqs = load_fasta(CACHE / 'universe.fa')
        packed, lens = [], []
        items = sorted(seqs.items(), key=lambda kv: len(kv[1]))
        i = 0
        with torch.no_grad():
            while i < len(items):
                batch, used = [], 0
                while i < len(items) and (len(batch) < 128 and used < 2048):
                    L = len(items[i][1]) + 1
                    if batch and used + L > 2048:
                        break
                    batch.append(items[i]); used += L; i += 1
                # ProtTrans convention: residue-split sequences (else T5's
                # sentencepiece swallows the whole string as ~2 tokens)
                tk_out = tok([' '.join(s) for _, s in batch], return_tensors='pt',
                             padding=True)
                # [v2-13 patch] attention_mask: pad tokens must not contaminate
                # residue embeddings (single per-query feature definition)
                out = enc(input_ids=tk_out.input_ids.cuda(),
                          attention_mask=tk_out.attention_mask.cuda()).last_hidden_state.half()
                for j, (_, s) in enumerate(batch):
                    packed.append(out[j, :len(s)].cpu().numpy())
                    lens.append(len(s))
                if len(packed) % 2048 < len(batch):
                    print(f'[prott5] {len(packed)}/{len(items)}', flush=True)
        flat = np.concatenate(packed, 0)
        np.savez(CACHE / 'prott5_perresidue.npz', flat=flat,
                 lens=np.array(lens), keys=np.array([s for _, s in items]))
        print('saved prott5_perresidue.npz', flat.shape)
        return

    if args.model == 'esm1b':
        _, out_tok = esm_extract('esm1b_t33_650M_UR50S', 33,
                                 args.toks_per_batch, per_token=True)
        out_dir = CACHE / 'esm1b_tok'
        out_dir.mkdir(exist_ok=True)
        for seq, v in out_tok.items():
            torch.save({'feature': v['feature'].half(),
                        'contacts': v['contacts'].half()}, out_dir / f'{seq}.pt')
        print('saved', len(out_tok), 'per-seq .pt to', out_dir)
        return

    cfg = {'esm2_3b': ('esm2_t36_3B_UR50D', 36, 'esm2_3b_L36.npz'),
           'esm1v': ('esm1v_t33_650M_UR90S_1', 33, 'esm1v_L33.npz')}[args.model]
    m, layer, out_name = cfg
    tp = 2048 if args.model == 'esm2_3b' else args.toks_per_batch
    out_mean, _ = esm_extract(m, layer, tp)
    np.savez(CACHE / out_name, **out_mean)
    print('saved', out_name, len(out_mean))


if __name__ == '__main__':
    main()
