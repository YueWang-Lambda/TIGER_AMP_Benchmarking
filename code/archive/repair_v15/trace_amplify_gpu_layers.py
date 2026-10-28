"""Trace same-checkpoint CPU/GPU differences without changing/saving weights."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/repair_v15/tf_gpu_repair'
tf.config.run_functions_eagerly(False)
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': True})
spec=importlib.util.spec_from_file_location('amptrace',ROOT/'baseline/sota_bench/retrain_amplify_regr_neg.py')
ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
model=ra.build_model()
model.load_weights(ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/workspace/runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s120.h5')
pep=pd.read_csv(ROOT/'reports/methods_package_S4_v1/2_external_testset/data/final11_peptides.csv')
x=ra.one_hot_padding(pep.sequence.tolist(),ra.MAXLEN)
dev='gpu' if tf.config.list_physical_devices('GPU') else 'cpu'
names=['Masking','Bidirectional-LSTM','Multi-Head-Attention','Attention','Output']
trace=tf.keras.Model(model.input,[model.get_layer(k).output for k in names])
vals=trace.predict(x,batch_size=32,verbose=0)
np.savez_compressed(OUT/f'amplify_layer_trace_{dev}.npz',**dict(zip(names,vals)))
(OUT/f'amplify_layer_trace_{dev}.json').write_text(json.dumps({'tf32':tf.config.experimental.tensor_float_32_execution_enabled(),'nvidia_tf32_override':os.getenv('NVIDIA_TF32_OVERRIDE'),'device':dev,'n':len(x)}))
print(dev,vals[-1].reshape(-1)[:5],flush=True)
