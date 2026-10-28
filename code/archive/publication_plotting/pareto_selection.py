"""Use the PCC-selected training condition for the main RSE cost plot."""
import numpy as np
import pandas as pd

def select_rse_at_pcc(rse_conditions, pcc_best):
    keys=['method','condition']
    assert not pcc_best.method.duplicated().any()
    assert not rse_conditions.duplicated(keys).any()
    selection=pcc_best[keys+['mean','n_defined_conditions']].rename(columns={'mean':'selection_PCC_mean','n_defined_conditions':'n_defined_PCC_conditions'})
    selected=selection.merge(rse_conditions,on=keys,how='left',validate='one_to_one')
    assert len(selected)==len(pcc_best) and selected.metric.eq('RSE').all()
    assert np.isfinite(selected['mean']).all() and np.isfinite(selected.selection_PCC_mean).all()
    selected['n_defined_conditions']=selected.method.map(rse_conditions.groupby('method')['mean'].count())
    selected['selection_metric']='PCC'
    selected['selection']='Condition selected by maximum defined ten-seed macro PCC; display corresponding RSE; descriptive test-set selection'
    return selected

def cost_frontier(data, time_col, metric):
    valid=data[np.isfinite(data['mean'])].copy();keep=[]
    for i,row in valid.iterrows():
        weak=valid['mean']>=row['mean'] if metric=='PCC' else valid['mean']<=row['mean']
        strict=valid['mean']>row['mean'] if metric=='PCC' else valid['mean']<row['mean']
        dominated=(weak&(valid[time_col]<=row[time_col])&(strict|(valid[time_col]<row[time_col]))).any()
        if not dominated:keep.append(i)
    return valid.loc[keep].sort_values(time_col)

def update_cost_selection(directory):
    from pathlib import Path
    directory=Path(directory)
    pcc=pd.read_csv(directory/'cost_best_PCC.csv',float_precision='round_trip')
    conditions={m:pd.read_csv(directory/f'cost_conditions_{m}.csv',float_precision='round_trip') for m in ['PCC','RSE']}
    selected=select_rse_at_pcc(conditions['RSE'],pcc)
    selected.to_csv(directory/'cost_best_RSE.csv',index=False)
    for tag,col in [('new_structures','time_s_with_struct'),('structures_ready','time_s_no_struct')]:
        cost_frontier(selected,col,'RSE').to_csv(directory/f'pareto_RSE_{tag}.csv',index=False)
        for metric in ['PCC','RSE']:
            # The supplementary plot still shows both conditions and their full frontier.
            cost_frontier(conditions[metric],col,metric).to_csv(directory/f'pareto_conditions_{metric}_{tag}.csv',index=False)
    return selected
