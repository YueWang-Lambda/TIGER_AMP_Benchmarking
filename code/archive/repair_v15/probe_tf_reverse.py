"""Isolate reverse/padding behavior in the installed TF/Keras GPU backend."""
from pathlib import Path
import importlib.util,json
import numpy as np,pandas as pd,tensorflow as tf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/repair_v15/tf_gpu_repair'
tf.config.run_functions_eagerly(False)
result={}
for depth in [20,512]:
 x=np.arange(32*50*depth,dtype=np.float32).reshape(32,50,depth)
 lengths=np.arange(12,44,dtype=np.int32)
 expected=x.copy()
 for i,n in enumerate(lengths):expected[i,:n]=x[i,:n][::-1]
 @tf.function
 def rev(t):return tf.reverse_sequence(t,lengths,seq_axis=1,batch_axis=0)
 with tf.device('/GPU:0'):
  eager=tf.reverse_sequence(x,lengths,seq_axis=1,batch_axis=0).numpy()
  graph=rev(x).numpy()
 result[str(depth)]={'reverse_eager_max':float(abs(expected-eager).max()),'reverse_graph_max':float(abs(expected-graph).max())}
spec=importlib.util.spec_from_file_location('ampdebug',ROOT/'baseline/sota_bench/retrain_amplify_regr_neg.py')
ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
model=ra.build_model();model.load_weights(ROOT/'artifacts/repair_v15/external_runs/AMPlify/seed120/workspace/runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s120.h5')
seq=pd.read_csv(ROOT/'runs_review/final11_peptides.csv').sequence.tolist();x=ra.one_hot_padding(seq,50).astype(np.float32)
layer=model.get_layer('Bidirectional-LSTM')
mask=x.any(-1)
for start in [0,32,64,128]:
 with tf.device('/CPU:0'):cpu=layer(tf.constant(x[start:start+32]),mask=tf.constant(mask[start:start+32]),training=False).numpy()
 with tf.device('/GPU:0'):gpu=layer(tf.constant(x[start:start+32]),mask=tf.constant(mask[start:start+32]),training=False).numpy()
 result[f'eager_layer_{start}']={'max':float(abs(cpu-gpu).max()),'fwd_max':float(abs(cpu[:,:,:512]-gpu[:,:,:512]).max()),'bwd_max':float(abs(cpu[:,:,512:]-gpu[:,:,512:]).max())}
(OUT/'reverse_probe.json').write_text(json.dumps(result,indent=2));print(result)
