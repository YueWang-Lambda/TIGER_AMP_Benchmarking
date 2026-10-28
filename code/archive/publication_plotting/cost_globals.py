"""Read only accepted timing coordinates with unified V16 TIGER-single; the notebook never launches timing."""
assert json.loads((OUT/'cost_acceptance.json').read_text())['status']=='PASS'
cost=pd.read_csv(OUT/'cost_timing_v15.csv')
control_cost={m:pd.read_csv(OUT/f'cost_conditions_{m}.csv') for m in ['PCC','RSE']}
control_best={m:pd.read_csv(OUT/f'cost_best_{m}.csv') for m in ['PCC','RSE']}
control_fronts={m:{tag:pd.read_csv(OUT/f'pareto_{m}_{tag}.csv') for tag in ['new_structures','structures_ready']} for m in ['PCC','RSE']}
METHOD_CATEGORY=CATEGORIES
CATEGORY_ORDER=['plm_finetune','frozen_plm','generative_lm','dl_sequence','dl_physicochemical','dl_sequence_physicochemical','multimodal_regression','multimodal_metric_learning']
CATEGORY_LABELS={'plm_finetune':'PLM fine-tuning','frozen_plm':'Frozen PLM + head','generative_lm':'Generative LM scorer','dl_sequence':'Sequence DL (no PLM)','dl_physicochemical':'Physicochemical models (no PLM)','dl_sequence_physicochemical':'Sequence + physicochemical DL (no PLM)','multimodal_regression':'3D multimodal regression','multimodal_metric_learning':'Multimodal metric learning'}
FIG5_CATEGORY_COLORS=CATEGORY_COLORS
FIG5_CATEGORY_TEXT_COLORS={'plm_finetune':'#5B478C','frozen_plm':'#287538','generative_lm':'#A54F00','dl_sequence':'#173D63','dl_physicochemical':'#006F6C','dl_sequence_physicochemical':'#88406F','multimodal_regression':'#202020','multimodal_metric_learning':'#202020'}
FIG5_THEME=dict(ink='#202020',muted='#8C949D',faded_text='#AEB4BA',frontier='#454545',green='#E4F9DF',gray='#FAFAFA')
method_marker=marker
noneg_display_name=display_name
assert set(cost.method)==set(CATEGORIES) and len(cost)==23
for metric in ['PCC','RSE']:
    assert len(control_cost[metric])==46 and len(control_best[metric])==23
    assert control_cost[metric].groupby('method').time_s_with_struct.nunique().eq(1).all()


# Main RSE uses exactly the PCC-selected condition; both-condition plots retain all observations.
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

control_best['RSE']=update_cost_selection(OUT)
control_fronts['RSE']={tag:pd.read_csv(OUT/f'pareto_RSE_{tag}.csv') for tag in ['new_structures','structures_ready']}
control_condition_fronts={m:{tag:pd.read_csv(OUT/f'pareto_conditions_{m}_{tag}.csv') for tag in ['new_structures','structures_ready']} for m in ['PCC','RSE']}
assert control_best['RSE'].set_index('method').condition.equals(control_best['PCC'].set_index('method').condition)
