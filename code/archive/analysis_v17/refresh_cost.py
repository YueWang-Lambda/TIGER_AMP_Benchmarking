"""Create V15 cost coordinates only from fully accepted new timing measurements."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from reconstruct import ROOT as R,OUT as O,WORK,FIXTURE,OLD
T=WORK/'timing'
b=pd.read_csv(O/'benchmark_summary.csv');methods=set(b.method)
rows=[];environments=[]
for method in sorted(methods):
    p=T/method/'result.json' if method=='TIGER-pair' and not FIXTURE else O/f'source_snapshot/timing/{method}.json';r=json.loads(p.read_text())
    assert r['status']=='PASS' and not r.get('acceptance_only') and r['method']==method and r['seed']==123
    assert len(r['repeats_s'])==20 and len(r['checks'])==21 and all(c['n_pair']==2211 and c['finite'] and c['independent_pair_algebra'] for c in r['checks'])
    assert np.isfinite(r['repeats_s']).all() and min(r['repeats_s'])>0
    rows.append({'method':method,'time_s_no_struct':np.median(r['repeats_s']),'time_min':min(r['repeats_s']),'time_max':max(r['repeats_s']),'device':r['device'],'seed':123,'n_repeats':20,'max_pair_diff':max(x['max_abs_pair_diff'] for x in r['checks'])})
    environments.append({'method':method,'main_python':r['python'],'device':r['device'],
        'resident_feature_child':{'ANIA':'/home/ubuntu/anaconda3/envs/ania/bin/python; R CGR',
                                  'GGN-ABPMIC':'/home/ubuntu/anaconda3/envs/ampacti/bin/python; descriptors',
                                  'EIPpred':'/home/ubuntu/anaconda3/envs/ampacti/bin/python; descriptors'}.get(method,''),
        'additional_executable':'/home/ubuntu/anaconda3/envs/dsspbio/bin/mkdssp' if method=='GGN-ABPMIC' else ''})
    d=O/'source_snapshot/timing';d.mkdir(exist_ok=True)
    if p!=d/(method+'.json'):shutil.copy2(p,d/(method+'.json'))
pd.DataFrame(environments).to_csv(O/'timing_environment_records.csv',index=False)
cost=pd.DataFrame(rows);structure=pd.read_csv(O/'structure_timing_paired_source_data.csv')
increment=float(structure.total_seconds.median()*208)
assert abs(increment-328.12)<1e-7
cost['structure_increment_s']=np.where(cost.method.isin(['TIGER-pair','TIGER-single','GGN-ABPMIC']),increment,0.)
cost['time_s_with_struct']=cost.time_s_no_struct+cost.structure_increment_s
cost.to_csv(O/'cost_timing_v15.csv',index=False)
for metric in ['PCC','RSE']:
    data=b[b.metric==metric].merge(cost,on='method',validate='many_to_one')
    data.to_csv(O/f'cost_conditions_{metric}.csv',index=False)
    chosen=[]
    for method,g in data.groupby('method'):
        valid=g[g['mean'].notna()].sort_values('mean',ascending=metric=='RSE')
        assert len(valid)>0
        row=valid.iloc[0].copy();row['n_defined_conditions']=len(valid);row['selection']='test-set descriptive; all-ten-seed mean'
        chosen.append(row)
    best=pd.DataFrame(chosen);best.to_csv(O/f'cost_best_{metric}.csv',index=False)
    for col,tag in [('time_s_with_struct','new_structures'),('time_s_no_struct','structures_ready')]:
        keep=[]
        for i,r in best.iterrows():
            better=(best['mean']>=r['mean']) if metric=='PCC' else (best['mean']<=r['mean'])
            strict=(best['mean']>r['mean']) if metric=='PCC' else (best['mean']<r['mean'])
            dominated=(better&(best[col]<=r[col])&(strict|(best[col]<r[col]))).any()
            if not dominated:keep.append(i)
        best.loc[keep].sort_values(col).to_csv(O/f'pareto_{metric}_{tag}.csv',index=False)
(O/'cost_acceptance.json').write_text(json.dumps({'status':'PASS','revision':'V17 unsigned-20000 pair and ensemble retimed; 22 other benchmark methods retained','methods':23,'timing_seed':123,'warmups':1,'repeats':20,'shared_condition_coordinates':'with-negative seed123 predictor; identical architecture and parameter count, not identical trained weights','structure_serial_estimate_s':increment,'structure_timing_source':'V14 retained measurements on 30 peptides; 208 times median paired-stage sum; not remeasured','metrics':['PCC','RSE'],'x_axis_minimum':.001,'canvas_mm':[325.12,190.5],'AMPLiT_PCC':'Only no-negative macro is defined; main display is best available and annotated, not a complete two-condition maximum'},indent=2))
cost.to_csv(O/'cost_timing_v17.csv',index=False)
print(cost[['method','time_s_no_struct','time_s_with_struct']].to_string(index=False))

addon=O/'TIGER_ensemble_addendum'
for fn in ['result.json','repeats.csv']:
    shutil.copy2((OLD/'TIGER_ensemble_addendum/timing' if FIXTURE else T/'TIGER-ensemble')/fn,addon/'timing'/fn)

import importlib.util
spec=importlib.util.spec_from_file_location('pareto_selection',O/'pareto_selection.py')
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)
selection.update_cost_selection(O)
p=O/'cost_acceptance.json';audit=json.loads(p.read_text());audit['validation_only']=FIXTURE;audit['main_RSE_condition_selection']='maximum defined ten-seed mean PCC; same condition as PCC plot'
audit['supplementary_frontier']='all displayed conditions';p.write_text(json.dumps(audit,indent=2))
