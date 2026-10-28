"""Multi-batch/order value gate on AMPlify GPU execution modes."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['arithmetic_off','meta_off','eager'],required=True);a=ap.parse_args()
if a.mode=='meta_off':tf.config.optimizer.set_experimental_options({'disable_meta_optimizer':True})
tf.config.run_functions_eagerly(a.mode=='eager')
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/repair_v15/tf_gpu_repair'
tf.keras.utils.set_random_seed(987654)
spec=importlib.util.spec_from_file_location('ampstable',ROOT/'baseline/sota_bench/retrain_amplify_regr_neg.py');ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
model=ra.build_model();model.load_weights(ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/workspace/runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s120.h5')
pep=pd.read_csv(ROOT/'runs_review/final11_peptides.csv');x=ra.one_hot_padding(pep.sequence.tolist(),50)
cpu=pd.read_csv(ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/predictions.csv').set_index('sequence').log10mic.reindex(pep.sequence).to_numpy()
result={'mode':a.mode,'checks':[]}
orders=[np.arange(208),np.arange(208),np.arange(207,-1,-1),np.random.RandomState(120).permutation(208)]
for k,order in enumerate(orders):
 t=time.perf_counter();y=model.predict(x[order],batch_size=32,verbose=0).ravel()*9-4
 row={'order':k,'max_log10mic_diff':float(abs(cpu[order]-y).max()),'count_over_1e4':int((abs(cpu[order]-y)>1e-4).sum()),'seconds':time.perf_counter()-t};result['checks'].append(row);print(row,flush=True)
result['pass']=all(r['max_log10mic_diff']<=1e-4 for r in result['checks'])
(OUT/f'amplify_stability_{a.mode}.json').write_text(json.dumps(result,indent=2))
