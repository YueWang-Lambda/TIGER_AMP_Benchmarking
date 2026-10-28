"""Build isolated, identity-aligned external inputs after negative deduplication.

Sequence-derived frozen representations can be reused. Positional caches are
filtered with their original row manifest; corpus-fitted TF-IDF is refitted.
The original data, feature files and checkpoints are never overwritten here.
"""
import argparse
import pickle
import shutil
from common import *
import numpy as np
import pandas as pd

SOTA=ROOT/'baseline/sota_bench'
DEST=WORK/'external_inputs'


def link(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        assert target.is_symlink() and target.resolve()==Path(source).resolve(),target
    else:target.symlink_to(Path(source).resolve(),target_is_directory=Path(source).is_dir())


def corpus():
    tr=pd.read_csv(PKG/'4_S4_benchmark/data/train_restricted4929_tiger.csv')
    pos=tr.groupby('sequence',as_index=False).MIC_Escherichia_coli.mean()
    old=pd.read_csv(SOTA/'data_restricted/uniprot_negatives.csv')
    neg=pd.read_csv(WORK/'data/external_negatives_disjoint.csv')
    assert len(pos)==4929 and len(old)==7962 and len(neg)==7949
    assert not set(pos.sequence)&set(neg.sequence)
    return pos,old,neg


def core():
    DEST.mkdir(parents=True,exist_ok=True)
    pos,old,neg=corpus();seqs=pos.sequence.tolist()+neg.sequence.tolist()
    oldseqs=pos.sequence.tolist()+old.sequence.tolist()
    # Drop negative duplicates only. All positive rows and their labels survive.
    keep=np.r_[np.arange(len(pos)),len(pos)+np.flatnonzero(~old.sequence.isin(pos.sequence))]
    assert list(np.asarray(oldseqs)[keep])==seqs and len(set(seqs))==12878
    pd.DataFrame({'new_index':np.arange(len(keep)),'old_index':keep,'sequence':seqs}).to_csv(DEST/'corpus_row_identity.csv',index=False)
    np.save(DEST/'old_to_new_keep_indices.npy',keep)
    for folder in ['emb_cache','ania_cache','ggn_work','data_restricted']:
        (DEST/folder).mkdir(exist_ok=True)
    changed={'emb_cache':{'eippred_feat_corpus.csv','ampacti_feat_corpus.csv','amppredictor_graphs.pt'},
             'ania_cache':{'corpus_ania.csv'},'ggn_work':{'train.csv'},
             'data_restricted':{'uniprot_negatives.csv'}}
    for folder,excluded in changed.items():
        for src in (SOTA/folder).iterdir():
            if src.name not in excluded:link(src,DEST/folder/src.name)
    neg.to_csv(DEST/'data_restricted/uniprot_negatives.csv',index=False)
    info={}
    for filename in ['eippred_feat_corpus.csv','ampacti_feat_corpus.csv']:
        df=pd.read_csv(SOTA/'emb_cache'/filename)
        seqcol='sequence' if 'sequence' in df else 'Sequence'
        assert df[seqcol].tolist()==oldseqs,filename
        sub=df.iloc[keep].reset_index(drop=True)
        assert sub[seqcol].tolist()==seqs
        sub.to_csv(DEST/'emb_cache'/filename,index=False)
        info[filename]={'old_rows':len(df),'new_rows':len(sub),'source_sha256':sha(SOTA/'emb_cache'/filename)}
    ania=pd.read_csv(SOTA/'ania_cache/corpus_ania.csv')
    assert ania.Sequence.tolist()==oldseqs
    sub=ania.iloc[keep].reset_index(drop=True)
    truth=np.r_[np.log10(pos.MIC_Escherichia_coli),np.full(len(neg),4.)]
    assert np.allclose(sub['Log MIC Value'],truth,rtol=0,atol=1e-12)
    sub.to_csv(DEST/'ania_cache/corpus_ania.csv',index=False)
    info['ania']={'old_rows':len(ania),'new_rows':len(sub),'positive_labels_preserved':True}
    ggn=pd.read_csv(SOTA/'ggn_work/train.csv')
    # Original preparation is positive rows followed by structure-available negatives.
    assert ggn.iloc[:4929].sequence.tolist()==pos.sequence.tolist()
    assert np.allclose(ggn.iloc[:4929].value,np.log10(pos.MIC_Escherichia_coli),atol=1e-7)
    ggn_keep=np.r_[np.arange(4929),4929+np.flatnonzero(~ggn.iloc[4929:].sequence.isin(pos.sequence))]
    ggn_new=ggn.iloc[ggn_keep].reset_index(drop=True)
    assert ggn_new.sequence.is_unique and len(ggn)-len(ggn_new)==13
    ggn_new.to_csv(DEST/'ggn_work/train.csv',index=False)
    info['GGN-ABPMIC']={'old_rows':len(ggn),'new_rows':len(ggn_new),'removed_negative_rows':13,
                        'sequence_keyed_graph_cache_label_replayed_from_csv':True}
    # SenseXAMP statistics use a unique-sequence table. Removing negative copies
    # of positive sequences leaves the table and its normalization unchanged.
    old_sex=pos.sequence.tolist()+old.loc[old.sequence.str.len()>=6,'sequence'].tolist()
    new_sex=pos.sequence.tolist()+neg.loc[neg.sequence.str.len()>=6,'sequence'].tolist()
    assert list(dict.fromkeys(old_sex))==list(dict.fromkeys(new_sex))
    link(SOTA/'sensexamp_cache',DEST/'sensexamp_cache')
    info['SenseXAMP']={'old_negative_rows':len(old_sex)-4929,'new_negative_rows':len(new_sex)-4929,
                      'unique_descriptor_fit_sequences_unchanged':True,'label_scaler_refitted_in_driver':True}
    info['affected_external_methods']=21
    info['additional_with_negative_runs_excluding_already_registered_AMPLiT']=200
    info['input_hashes']={'positives':sha(PKG/'4_S4_benchmark/data/train_restricted4929_tiger.csv'),
                          'original_negatives':sha(SOTA/'data_restricted/uniprot_negatives.csv'),
                          'corrected_negatives':sha(WORK/'data/external_negatives_disjoint.csv')}
    dump(DEST/'core_inputs.json',{'status':'PASS',**info})
    print('External core inputs PASS',flush=True)


def amppredictor():
    import torch
    torch.set_num_threads(2)
    target=DEST/'emb_cache/amppredictor_graphs.pt'
    if target.exists():
        assert load(target.with_suffix('.identity.json'))['status']=='PASS';return
    pos,old,neg=corpus();keep=np.load(DEST/'old_to_new_keep_indices.npy')
    src=SOTA/'emb_cache/amppredictor_graphs.pt'
    items=torch.load(src,map_location='cpu',weights_only=False)
    assert len(items)==len(pos)+len(old)
    # The original driver creates this cache from corpus() in this exact order.
    # Independently check fingerprints and node features at all removed indices
    # against their corresponding retained positive copies before filtering.
    fp=np.load(SOTA/'emb_cache/amppredictor_fp_corpus.npz')
    seqs=pos.sequence.tolist()+old.sequence.tolist()
    for i in np.flatnonzero(old.sequence.isin(pos.sequence)):
        j=len(pos)+int(i);s=seqs[j];p=pos.sequence.tolist().index(s)
        assert torch.equal(items[j][0].x,items[p][0].x)
        assert np.array_equal(items[j][1].numpy(),fp[s])
    selected=[items[int(i)] for i in keep]
    torch.save(selected,target)
    dump(target.with_suffix('.identity.json'),{'status':'PASS','n_rows':len(selected),'source_sha256':sha(src),
         'row_manifest_sha256':sha(DEST/'corpus_row_identity.csv'),'duplicate_graphs_and_fingerprints_equal':True})
    print('AMPpredictor positional cache PASS',len(selected),flush=True)


def hyperamp():
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    pos,old,neg=corpus();target=DEST/'hyperamp_cache';target.mkdir(exist_ok=True)
    names=[f'p{i}' for i in range(len(pos))]+[f'n{i}' for i in range(len(neg))]
    seqs=pos.sequence.tolist()+neg.sequence.tolist()
    keep_old=np.flatnonzero(~old.sequence.isin(pos.sequence))
    src=np.load(SOTA/'hyperamp_cache/ankh_emb.npz')
    remap={k:k for k in src.files if k.startswith('p') or k.startswith('e')}
    remap.update({f'n{i}':f'n{j}' for i,j in enumerate(keep_old)})
    assert all(k in src.files for k in remap.values())
    np.savez(target/'ankh_emb.npz',**{k:src[v] for k,v in remap.items()})
    dump(target/'embedding_key_mapping.json',remap)
    tfidf_lookup={};k_idf_list=[]
    # Retain the registered HyperAMP implementation's lookup construction.
    for k in (2,3,4):
        vec=CountVectorizer(ngram_range=(k,k),analyzer='char');X=vec.fit_transform(seqs)
        words=vec.get_feature_names_out();tf=TfidfTransformer(use_idf=False).fit_transform(X)
        idf=TfidfTransformer(use_idf=True,smooth_idf=False).fit(X).idf_
        for r in range(X.shape[0]):
            row=X.getrow(r);tfr=tf.getrow(r)
            # Sparse extraction is algebraically identical to the dense row in
            # the original driver, while avoiding a corpus-sized dense array.
            vals=dict(zip(tfr.indices,tfr.data))
            tfidf_lookup[names[r]]={words[j]:float(vals[j]*idf[j]+1.) for j in row.indices}
        k_idf_list.append({k:{w:idf[j] for j,w in enumerate(words)}})
        print('HyperAMP refit TF-IDF k',k,flush=True)
    for fn,obj in [('kgram_tfidf.pkl',tfidf_lookup),('k_idf.pkl',k_idf_list)]:
        with (target/fn).open('wb') as f:pickle.dump(obj,f)
    dump(target/'identity.json',{'status':'PASS','n_train':len(seqs),'fit_sequences':seqs,
          'embedding_source_sha256':sha(SOTA/'hyperamp_cache/ankh_emb.npz'),
          'negative_sha256':sha(WORK/'data/external_negatives_disjoint.csv')})
    print('HyperAMP cache PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['core','amppredictor','hyperamp']);a=ap.parse_args()
    {'core':core,'amppredictor':amppredictor,'hyperamp':hyperamp}[a.stage]()
