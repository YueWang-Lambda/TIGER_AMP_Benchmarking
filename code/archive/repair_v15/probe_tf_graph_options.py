"""Isolate graph rewrite responsible for GPU masked-backward-LSTM divergence."""
from pathlib import Path
import importlib.util,json,time
import numpy as np,pandas as pd,tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/repair_v15/tf_gpu_repair'
tf.config.run_functions_eagerly(False)
spec=importlib.util.spec_from_file_location('ampopts',ROOT/'baseline/sota_bench/retrain_amplify_regr_neg.py')
ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
pep=pd.read_csv(ROOT/'runs_review/final11_peptides.csv');x=ra.one_hot_padding(pep.sequence.tolist(),50)
cpu=np.load(OUT/'amplify_layer_trace_cpu.npz')['Output']
results={}
base=tf.config.optimizer.get_experimental_options()
for mode,options in [('disable_layout',{'layout_optimizer':False}),
                     ('disable_arithmetic',{'arithmetic_optimization':False}),
                     ('disable_meta',{'disable_meta_optimizer':True})]:
 tf.keras.backend.clear_session()
 # Reset keys explicitly because set_experimental_options merges its argument.
 tf.config.optimizer.set_experimental_options(dict(base,layout_optimizer=True,arithmetic_optimization=True,disable_meta_optimizer=False))
 tf.config.optimizer.set_experimental_options(options)
 model=ra.build_model();model.load_weights(ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/workspace/runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s120.h5')
 t=time.perf_counter();y=model.predict(x,batch_size=32,verbose=0)
 results[mode]={'max_log10mic_diff':float(abs(cpu-y).max()*9),'seconds':time.perf_counter()-t,'options':tf.config.optimizer.get_experimental_options()}
 print(mode,results[mode],flush=True)
 (OUT/'graph_options_probe.json').write_text(json.dumps(results,indent=2))
