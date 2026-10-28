"""Isolated CPU diagnostic; never modifies campaign weights or predictions."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    out = ROOT/'artifacts/repair_v15/amplify_runtime_audit'
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
        TF_NUM_INTRAOP_THREADS=str(args.threads), TF_NUM_INTEROP_THREADS='1',
        TF_CPP_MIN_LOG_LEVEL='2')
    # Avoid the project's code package shadowing stdlib code during TF import.
    sys.path = [p for p in sys.path if p and Path(p).resolve() != ROOT]
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    src = ROOT/'baseline/sota_bench/retrain_amplify_regr_neg.py'
    spec = importlib.util.spec_from_file_location('amplify_runtime_driver', src)
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    tf.keras.utils.set_random_seed(987654)
    model = ra.build_model()
    mha = model.get_layer('Multi-Head-Attention')
    result = dict(tf_version=tf.__version__, threads=args.threads,
        gpu_devices=[str(x) for x in tf.config.list_physical_devices('GPU')],
        parameters=model.count_params(),
        layers=[{'name':x.name, 'parameters':x.count_params()} for x in model.layers],
        initial_mha_equal={k:bool(np.array_equal(mha.Wq.numpy(),getattr(mha,k).numpy()))
                           for k in ['Wk','Wv','Wo']},
        load_start=list(os.getloadavg()), formal_results_untouched=True)
    ckpt = ROOT/'runs_review/benchmark_restricted/sota_models/amplify_regr_neg/amplify_s123.h5'
    model.load_weights(ckpt)
    positive = pd.read_csv(ROOT/'reports/methods_package_S4_v1/4_S4_benchmark/data/train_restricted4929_tiger.csv').groupby('sequence',as_index=False).MIC_Escherichia_coli.mean()
    negative = pd.read_csv(ROOT/'artifacts/repair_v15/external_inputs/data_restricted/uniprot_negatives.csv')
    seqs = positive.sequence.tolist()+negative.sequence.tolist()
    targets = (np.r_[np.log10(positive.MIC_Escherichia_coli.to_numpy()),np.full(len(negative),4.)]+4.)/9.
    n = len(seqs); nval=int(n*.1)
    result.update(n_corpus=n,n_train=n-nval,n_val=nval,
        train_batches=math.ceil((n-nval)/32),validation_batches=math.ceil(nval/32))
    pep = pd.read_csv(ROOT/'runs_review/final11_peptides.csv').sequence.tolist()
    xext = ra.one_hot_padding(pep,ra.MAXLEN)
    model.predict(xext,verbose=0)
    inference=[]
    for _ in range(3):
        t=time.perf_counter(); model.predict(ra.one_hot_padding(pep,ra.MAXLEN),verbose=0)
        inference.append(time.perf_counter()-t)
    result['inference_208_seconds']=inference
    idx=np.random.RandomState(987654).permutation(n)[:320]
    x=ra.one_hot_padding([seqs[i] for i in idx],ra.MAXLEN); y=targets[idx]
    model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(.001))
    train_times=[]; val_times=[]
    class Timing(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None): self.epoch=epoch
        def on_train_batch_begin(self,batch,logs=None): self.t=time.perf_counter()
        def on_train_batch_end(self,batch,logs=None):
            if self.epoch>0:train_times.append(time.perf_counter()-self.t)
        def on_test_batch_begin(self,batch,logs=None): self.t=time.perf_counter()
        def on_test_batch_end(self,batch,logs=None):
            if self.epoch>0:val_times.append(time.perf_counter()-self.t)
    before=model.get_layer('Output').get_weights()[0].copy()
    hist=model.fit(x[:256],y[:256],batch_size=32,epochs=3,
        validation_data=(x[256:],y[256:]),callbacks=[Timing()],verbose=0)
    result.update(train_batch_seconds=train_times,val_batch_seconds=val_times,
        fit_loss=[float(v) for v in hist.history['loss']],
        output_weights_changed=bool(not np.array_equal(before,model.get_layer('Output').get_weights()[0])),
        all_losses_finite=bool(np.isfinite(hist.history['loss']).all()),
        estimated_epoch_seconds=float(np.median(train_times)*result['train_batches']+
                                      np.median(val_times)*result['validation_batches']))
    # Exercise the installed callback with known losses and explicit fake weights.
    class Dummy:
        stop_training=False
        epoch=0
        def get_weights(self):return [self.epoch]
        def set_weights(self,w):self.restored_epoch=w[0]
    dummy=Dummy(); cb=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        patience=50,min_delta=.001,restore_best_weights=True)
    cb.set_model(dummy);cb.on_train_begin()
    losses=[.006,.0048,.0045,.0040]+[.0039]*60
    for i,v in enumerate(losses):
        dummy.epoch=i+1;cb.on_epoch_end(i,{'val_loss':v})
        if dummy.stop_training:break
    result['early_stopping_control']=dict(stopped_epoch=i+1,
        restored_epoch=dummy.restored_epoch,qualified_best_loss=float(cb.best),
        absolute_min_loss=min(losses[:i+1]))
    result['load_end']=list(os.getloadavg())
    dest=out/f'probe_threads{args.threads}.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['train_batch_seconds','val_batch_seconds','layers']},indent=2))


if __name__=='__main__':main()
