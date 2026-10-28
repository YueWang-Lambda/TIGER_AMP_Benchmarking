"""Audit registered original corpus functions and actual pre-training arrays."""
import ast
from common import *
import numpy as np
import pandas as pd
from check_negative_precedence import capture_before

SOTA=ROOT/'baseline/sota_bench'
OUT=WORK/'negative_precedence_audit'


def original_function(path,name,*args):
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[fn],type_ignores=[])
    ns={'pd':pd,'np':np,'ROOT':ROOT,'TIGER':ROOT,'V2':ROOT,'SOTA':SOTA,'HERE':SOTA,
        'TRAIN_CSV':ROOT/'runs_review/benchmark_restricted/train_restricted4929_tiger.csv',
        'NEG_CSV':SOTA/'data_restricted/uniprot_negatives.csv','NEG_LOG10':4.}
    exec(compile(module,str(path),'exec'),ns)
    return ns[name](*args)


def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    def record(method,src,seq,y,boundary):
        d=pd.DataFrame({'sequence':list(seq),'target':np.asarray(y).reshape(-1)})
        assert np.isfinite(d.target).all()
        conflict=d.groupby('sequence').target.nunique().gt(1)
        n=int(conflict.sum());assert n==13,(method,n)
        d[d.sequence.isin(conflict[conflict].index)].to_csv(OUT/f'{method}_corpus_conflicts.csv',index=False)
        rows.append({'method':method,'source':str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
             'source_sha256':sha(src),'n_rows':len(d),'conflicting_sequences':n,'inspection_boundary':boundary,
             'training_executed':False})
    for method,fn in [('EIPpred','retrain_eippred_neg.py'),('AMPActiPred','retrain_ampacti_neg.py'),
                       ('AMPpredictor','retrain_amppredictor_neg.py')]:
        src=SOTA/fn;d=original_function(src,'corpus',False)
        record(method,src,d.sequence,d.MIC_Escherichia_coli,'original corpus(False), row-wise targets retained by fit/DS')
    src=SOTA/'retrain_ania_neg.py';d=pd.read_csv(SOTA/'ania_cache/corpus_ania.csv')
    record('ANIA',src,d.Sequence,d['Log MIC Value'],'archived feature/target CSV read row-wise by extract_cgr_features_and_target_for_dl')
    for method,fn in [('MBC-Attention','retrain_mbc_attention_neg.py'),('HyperAMP','retrain_hyperamp_neg.py'),
                       ('SenseXAMP','retrain_sensexamp_neg.py'),('AMPlify','retrain_amplify_regr_neg.py')]:
        src=SOTA/fn;d=original_function(src,'load_universe',False)
        if method=='MBC-Attention':seq,y=d.SEQUENCE,d.EC_pMIC
        elif method=='HyperAMP':_,seq,y=d
        elif method=='SenseXAMP':seq,y,_=d
        else:seq,y=d
        record(method,src,seq,y,'original load_universe(False), subsequent row-index split preserves labels')
    for method in ['LLAMP','BERT-AmPEP60','Deep-PIE']:
        src=SOTA/'train_sota_bench_neg.py';seq,y=original_function(src,'load_universe','with_neg')
        record(method,src,seq,y,'original load_universe(with_neg), DS stores sequence/label lists')
    # Original mains execute their input and split code, then raise before fitting.
    src=SOTA/'retrain_das_regressor.py';d=capture_before(src,'model = BiLSTMRegressor()',['--seed','123'])
    record('Das',src,d['seqs'],d['y'],'actual main before model construction; split audit in verification.json')
    src=SOTA/'retrain_amplit_regressor_fix.py';d=capture_before(src,"cache = Path('/tmp/amplit_feats_noneg.npz'",['--seed','123','--variant','B'])
    record('AMPLiT',src,d['seqs'],d['y'],'actual main before feature cache; labels later z-scaled without deduplication')
    src=SOTA/'retrain_hydramp_regressor_neg.py';d=capture_before(src,'from sklearn.model_selection import KFold',['--seed','123'])
    seq=[s for s,r in zip(d['pos'].sequence,d['pos_rows']) if r is not None]+[s for s,r in zip(d['neg'].sequence,d['neg_rows']) if r is not None]
    record('HydrAMP',src,seq,d['y_all'],'actual encoded x_all/y_all before KFold')
    src=SOTA/'retrain_deep_amp.py';d=capture_before(src,'Xt = torch.from_numpy(X)',['--seeds','123'])
    seq=[s for s,r in zip(d['pos_seqs'],d['rows_p']) if r is not None]+[s for s,r in zip(d['neg_seqs'],d['rows_n']) if r is not None]
    record('DeepAMP',src,seq,d['y'],'actual filtered X/y before tensor dataset')
    src=SOTA/'retrain_ampgen_neg.py';d=capture_before(src,'Xtr, Xva, ytr, yva = train_test_split',['--seed','123'])
    record('AMPGen',src,d['corpus'].sequence,d['y'],'actual X/y before train_test_split')
    src=SOTA/'retrain_tsamp_neg.py';d=capture_before(src,'idx = torch.randperm',['--seed','123'])
    record('tsAMP',src,d['corpus'].sequence,d['y'],'actual X/y before train split and augmentation')
    src=ROOT.parents[1]/'baseline/sota_bench/retrain_evo_strict.py';d=capture_before(src,"df = pd.DataFrame({'Sequence': seqs})",['--seed','123'])
    record('EvoGradient',src,d['seqs'],d['reg'],'actual regression targets; unique sequence split retains both labels within partition')
    src=SOTA/'apex-main/train_apex_lrfix_neg.py';sys.path.insert(0,str(src.parent))
    d=capture_before(src,'rng = np.random.default_rng(42)',['--mode','with_neg','--seeds','1','--seed-base','123'])
    record('APEX',src,d['seqs'],d['y'],'actual activity-scale y before split')
    src=SOTA/'AMP-Designer/train_ampd_restricted_neg.py';sys.path.insert(0,str(src.parent))
    d=capture_before(src,'tokenizer = BertTokenizer',['--mode','with_neg','--seeds','1','--seed-base','123'])
    record('AMP-Designer',src,d['seqs'],d['y'],'actual sequence/target lists before tokenizer')
    src=SOTA/'retrain_ggn_neg.py';d=pd.read_csv(SOTA/'ggn_work/train.csv')
    record('GGN-ABPMIC',src,d.sequence,d.value,'actual ProteinGraphDataset independently verified in verification.json')
    assert len(rows)==21 and len({r['method'] for r in rows})==21
    pd.DataFrame(rows).to_csv(OUT/'all_external_corpus_checks.csv',index=False)
    dump(OUT/'all_external_corpus_checks.json',{'status':'PASS','methods':rows,
         'boundary_note':'Three methods have actual-fit/dataset/split captures; others have original corpus/pre-training-array execution plus downstream row-wise-label code inspection. No external training is performed.'})
    print('All 21 original external corpus routes contain conflicting labels',flush=True)


if __name__=='__main__':main()
