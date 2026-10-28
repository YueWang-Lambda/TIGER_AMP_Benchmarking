"""Read-only audit: inspect actual pre-training inputs, never fit a model."""
import ast
import importlib.util
import inspect
from common import *
import numpy as np
import pandas as pd


class Captured(Exception):
    pass


def capture_before(path,needle,argv):
    """Execute the original main only to a known training boundary.

    Insert an exception immediately before that statement. All preceding input
    selection/splitting is the original code, not an independently guessed recipe.
    """
    text=path.read_text();tree=ast.parse(text);found=0;result={}
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name=='main':
            new=[]
            for stmt in fn.body:
                if needle in ast.get_source_segment(text,stmt):
                    new.append(ast.parse('_capture(locals())').body[0]);found+=1
                new.append(stmt)
            fn.body=new
    assert found==1,(path,needle,found)
    def cap(values):result.update(values);raise Captured()
    namespace={'__name__':'_readonly_training_capture','__file__':str(path),'_capture':cap}
    ast.fix_missing_locations(tree);exec(compile(tree,str(path),'exec'),namespace)
    old=sys.argv;sys.argv=[str(path),*argv]
    try:
        namespace['main']()
        raise RuntimeError('Training boundary was not intercepted')
    except Captured:pass
    finally:sys.argv=old
    return result


def main():
    out=WORK/'negative_precedence_audit';out.mkdir(exist_ok=True)
    original=ARCHIVE/'code/train.py'
    before=original.read_text();now=(ROOT/'code/train.py').read_text()
    expression='keep = ~neg_payload["neg_rows"].sequence.isin(set(rows.sequence))'
    assert expression in before and expression in now
    config=load(WORK/'configs/S4_seed123.json')
    print('TIGER original positive-precedence filter confirmed',flush=True)
    results={'TIGER':{'positive_precedence_exists_before_repair':True,
             'expression':expression,'negative_csv':config['neg_csv']}}
    # EIPpred reaches clf.fit with both observations: capture its exact X/y.
    src=ROOT/'baseline/sota_bench/retrain_eippred_neg.py'
    e=capture_before(src,'clf.fit(X, y)',['--seed','123'])
    t=e['tr'];mask=t.sequence.duplicated(keep=False)
    coll=t.loc[mask].copy();coll['actual_fit_target']=e['y'][mask]
    groups=coll.groupby('sequence');assert len(groups)==13 and (groups.actual_fit_target.nunique()==2).all()
    for seq,g in t[mask].groupby('sequence'):
        ii=g.index.to_numpy();assert len(ii)==2 and np.array_equal(e['X'][ii[0]],e['X'][ii[1]])
    coll.to_csv(out/'EIPpred_actual_fit_conflicting_rows.csv',index=False)
    results['EIPpred']={'captured_before':'clf.fit(X, y)','fit_rows':len(e['X']),
         'conflicting_sequences_in_fit':len(groups),'identical_features_with_distinct_targets':True,
         'source_sha256':sha(src),'training_executed':False}
    # Das: preserve the original fixed split, then inspect actual optimizer rows.
    src=ROOT/'baseline/sota_bench/retrain_das_regressor.py'
    d=capture_before(src,'model = BiLSTMRegressor()',['--seed','123'])
    full=pd.DataFrame({'sequence':d['seqs'],'log10mic':d['y'],'partition':'unused'})
    full.loc[d['tr_i'],'partition']='train';full.loc[d['va_i'],'partition']='validation'
    mask=full.sequence.duplicated(keep=False);full.loc[mask].to_csv(out/'Das_actual_split_conflicting_rows.csv',index=False)
    both=full[full.partition=='train'].groupby('sequence').log10mic.nunique().gt(1)
    results['Das']={'captured_before':'BiLSTMRegressor construction','all_rows':len(full),'fit_rows':len(d['tr_i']),
          'conflicting_sequences_in_corpus':int(full[mask].sequence.nunique()),
          'conflicting_sequences_both_in_training':int(both.sum()),'source_sha256':sha(src),'training_executed':False}
    # GGN caches geometry by sequence, but the dataset reads each row's label.
    src=ROOT/'baseline/sota_bench/GGN-ABPMIC/myData.py'
    spec=importlib.util.spec_from_file_location('readonly_ggn_data',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data_root=ROOT/'baseline/sota_bench/ggn_work'
    args=type('Args',(),{'train_csv':str(data_root/'train.csv'),'val_csv':str(data_root/'ext.csv'),
        'pdb_dir':str(data_root/'pdb'),'prottrans_dir':str(data_root/'prottrans'),
        'dssp_dir':str(data_root/'dssp'),'device':'cpu'})()
    prior=os.environ.pop('GGN_GRAPH_CACHE',None)
    try:dataset=mod.ProteinGraphDataset(args)
    finally:
        if prior is not None:os.environ['GGN_GRAPH_CACHE']=prior
    frame=pd.DataFrame({'sequence':dataset.sequences,'actual_dataset_label':dataset.values})
    mask=frame.sequence.duplicated(keep=False);frame[mask].to_csv(out/'GGN_actual_dataset_conflicting_rows.csv',index=False)
    assert frame[mask].sequence.nunique()==13
    assert (frame[mask].groupby('sequence').actual_dataset_label.nunique()==2).all()
    results['GGN-ABPMIC']={'actual_dataset_rows':len(dataset),'conflicting_sequences_in_dataset':13,
                         'row_labels_retained_by_dataset':True,'source_sha256':sha(src),'training_executed':False}
    dump(out/'verification.json',{'status':'PASS','results':results,
        'conclusion':'TIGER enforces positive precedence; external counterexamples retain conflicting labels. Do not generalize the TIGER guard to all drivers.'})
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
