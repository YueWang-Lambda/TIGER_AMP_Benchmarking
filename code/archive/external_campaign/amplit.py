"""One formal AMPLiT regression adaptation with a persisted training encoder.

Word2Vec is fitted only on the training subset; query transform never fits.
Descriptor values are aligned by saved descriptor names, including when short
sequences need reduced-lag fallback. Historical A/B/C scores are not selected.
"""
import argparse
import fcntl
import time
from common import *
import numpy as np
import pandas as pd

AA='ACDEFGHIKLMNPQRSTVWY'
SCHEMA='amplit_fixed_w2v_named_descriptors_v15_2'


def descriptor_values(seq):
    from propy import PyPro
    obj=PyPro.GetProDes(seq);length=len(seq)
    calls=[('GetAAComp',{}),('GetDPComp',{}),('GetMoreauBrotoAuto',{}),('GetMoranAuto',{}),
           ('GetGearyAuto',{}),('GetCTD',{}),('GetPAAC',{'lamda':min(10,length-1),'weight':.05}),
           ('GetAPAAC',{'lamda':min(10,length-1),'weight':.5}),
           ('GetSOCN',{'maxlag':min(45,length-1)}),('GetQSO',{'maxlag':min(30,length-1),'weight':.1})]
    result={};undefined=[]
    for name,kwargs in calls:
        try:
            values=getattr(obj,name)(**kwargs)
            result.update({k:float(v) for k,v in values.items() if np.isfinite(float(v))})
            if any(not np.isfinite(float(v)) for v in values.values()):undefined.append(name)
        except ZeroDivisionError:undefined.append(name)
    if not result:raise ValueError(f'No valid descriptors for {seq}')
    # Constant-property correlations are mathematically undefined; retain all
    # other valid descriptor families and fill only absent schema keys with 0.
    return result,','.join(undefined) or 'valid'


def feature_schema():
    vals,_=descriptor_values(AA*3)
    names=list(vals)
    if len(names)>1550:raise ValueError(f'Descriptor schema exceeds architecture width: {len(names)}')
    return names


def static_features(seqs,names):
    onehot=np.zeros((len(seqs),50,20),dtype=np.float32)
    desc=np.zeros((len(seqs),50,31),dtype=np.float32)
    fallbacks={};missing=[]
    for i,s in enumerate(seqs):
        if not s or any(a not in AA for a in s):raise ValueError('Noncanonical sequence')
        for j,a in enumerate(s[:50]):onehot[i,j,AA.index(a)]=1
        values,lag=descriptor_values(s)
        fallbacks[str(lag)]=fallbacks.get(str(lag),0)+1
        # Match semantic keys; different fallback dictionary layouts must not
        # silently shift subsequent feature columns.
        desc[i].reshape(-1)[:len(names)]=[float(values.get(k,0.)) for k in names]
        missing.append(sum(k not in values for k in names))
        if len(seqs)>1000 and (i+1)%1000==0:print('AMPLiT descriptors',i+1,'/',len(seqs),flush=True)
    return onehot,desc,{'fallback_counts':fallbacks,'max_missing_descriptors':max(missing,default=0)}


def training_data(condition):
    pos=pd.read_csv(PKG/'4_S4_benchmark/data/train_restricted4929_tiger.csv')
    pos=pos.groupby('sequence',as_index=False).MIC_Escherichia_coli.mean()
    assert len(pos)==4929
    seqs=pos.sequence.tolist();y=np.log10(pos.MIC_Escherichia_coli.to_numpy(float)).astype(np.float32)
    rng=np.random.default_rng(42);pp=rng.permutation(len(pos));nv=int(.1*len(pos))
    train=pp[nv:].tolist();valid=pp[:nv].tolist()
    if condition=='with_negatives':
        neg=pd.read_csv(WORK/'data/external_negatives_disjoint.csv')
        assert len(neg)==7949 and not set(neg.sequence)&set(seqs)
        seqs+=neg.sequence.tolist();y=np.concatenate([y,np.full(len(neg),4.,dtype=np.float32)])
        pn=np.random.default_rng(4242).permutation(len(neg));n=int(.1*len(neg))
        train+=(pn[n:]+len(pos)).tolist();valid+=(pn[:n]+len(pos)).tolist()
    elif condition!='without_negatives':raise ValueError(condition)
    return seqs,y,np.asarray(train),np.asarray(valid)


def cached_static(condition):
    seqs,y,tr,va=training_data(condition);names=feature_schema()
    ident={'schema':SCHEMA,'condition':condition,'sequence_order':seqs,'descriptor_names':names,'code_sha256':sha(__file__)}
    key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    path=WORK/'amplit_features'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not path.exists():
            x1,x3,stats=static_features(seqs,names)
            temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with temp.open('wb') as out:np.savez_compressed(out,onehot=x1,descriptors=x3)
            os.replace(temp,path);dump(path.with_suffix('.json'),{'identity':ident,'statistics':stats})
        z=np.load(path);x1,x3=z['onehot'],z['descriptors']
    return seqs,y,tr,va,names,x1,x3


def fit_word_vectors(seqs,seed):
    from gensim.models import Word2Vec
    # A stable hash avoids Python process hash randomization changing initialization.
    def stable_hash(word):return int.from_bytes(hashlib.sha256(word.encode()).digest()[:4],'little')
    model=Word2Vec(sentences=[list(s) for s in seqs],vector_size=100,min_count=1,
                   seed=seed,workers=1,hashfxn=stable_hash)
    vectors=np.stack([model.wv[a] if a in model.wv else np.zeros(100) for a in AA]).astype(np.float32)
    return vectors


def embed(seqs,vectors):
    x=np.zeros((len(seqs),50,100),dtype=np.float32)
    for i,s in enumerate(seqs):
        for j,a in enumerate(s[:50]):x[i,j]=vectors[AA.index(a)]
    return x


def architecture(device):
    sys.path.insert(0,str(ROOT/'baseline/sota_bench'))
    from retrain_amplit_regressor import AMPLiTRegressor
    import torch
    model=AMPLiTRegressor().to(device)
    model.eval()
    with torch.no_grad():model(torch.zeros(1,50,20,device=device),torch.zeros(1,50,100,device=device),torch.zeros(1,50,31,device=device))
    return model


class Predictor:
    def __init__(self,path,device='cuda:0'):
        import torch
        self.device=torch.device(device)
        ck=torch.load(Path(path)/'model.pt',map_location='cpu',weights_only=False)
        self.model=architecture(self.device);self.model.load_state_dict(ck['model']);self.model.eval()
        self.vectors=np.load(Path(path)/'word_vectors.npy');self.spec=load(Path(path)/'encoder.json')
        self.mean,self.std=ck['label_mean'],ck['label_std']

    def predict(self,seqs):
        import torch
        x1,x3,_=static_features(seqs,self.spec['descriptor_names']);x2=embed(seqs,self.vectors)
        ys=[]
        with torch.inference_mode():
            for i in range(0,len(seqs),256):
                xs=[torch.from_numpy(x[i:i+256]).to(self.device) for x in [x1,x2,x3]]
                ys.append(self.model(*xs).cpu().numpy())
        return np.concatenate(ys)*self.std+self.mean


def train(condition,seed,gpu,smoke=False):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(seed);np.random.seed(seed)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    out=(WORK/'amplit_smoke' if smoke else RUNS/'AMPLiT')/condition/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():raise RuntimeError('Completed AMPLiT run exists')
    start=time.time();seqs,y,tr,va,names,x1,x3=cached_static(condition)
    vectors=fit_word_vectors([seqs[i] for i in tr],seed);x2=embed(seqs,vectors)
    # This mathematical transform is independent of query batch and ordering.
    assert np.array_equal(embed([seqs[0]],vectors)[0],embed(seqs[:7],vectors)[0])
    mu,sd=float(y[tr].mean()),float(y[tr].std());assert np.isfinite(sd) and sd>0
    yz=(y-mu)/sd
    model=architecture(device)
    head=list(model.head.parameters())+list(model.cbam.parameters());ids={id(p) for p in head}
    trunk=[p for p in model.parameters() if id(p) not in ids]
    opt=torch.optim.AdamW([{'params':trunk,'lr':1e-4},{'params':head,'lr':1e-3}],weight_decay=1e-6)
    lossf=torch.nn.SmoothL1Loss();best=float('inf');stale=0;history=[];best_state=None
    def loss_batch(idx):
        xs=[torch.from_numpy(x[idx]).to(device) for x in [x1,x2,x3]]
        return lossf(model(*xs),torch.from_numpy(yz[idx]).to(device))
    for ep in range(1,(2 if smoke else 60)+1):
        model.train();perm=np.random.permutation(tr);losses=[]
        for i in range(0,len(perm),512):
            opt.zero_grad();loss=loss_batch(perm[i:i+512])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vs=[(float(loss_batch(va[i:i+512])),len(va[i:i+512])) for i in range(0,len(va),512)]
        vl=sum(v*n for v,n in vs)/len(va)
        if not np.isfinite(vl):raise ValueError('Nonfinite validation loss')
        history.append({'epoch':ep,'train_loss':float(np.mean(losses)),'val_loss':vl})
        print(condition,seed,'epoch',ep,'val_loss',vl,flush=True)
        if vl<best-1e-4:
            best,stale=vl,0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=7:break
    assert best_state is not None
    torch.save({'model':best_state,'label_mean':mu,'label_std':sd,'seed':seed,'condition':condition,'schema':SCHEMA},out/'model.pt')
    np.save(out/'word_vectors.npy',vectors)
    dump(out/'encoder.json',{'schema':SCHEMA,'descriptor_names':names,'AA':AA,'word2vec_seed':seed,'word2vec_workers':1,
        'fitted_sequences':[seqs[i] for i in tr],'missing_descriptor_policy':'name-aligned zero fill; lag capped at sequence length minus one; undefined constant-property descriptor families zero-filled',
        'architecture_source_sha256':sha(ROOT/'baseline/sota_bench/retrain_amplit_regressor.py')})
    dump(out/'split.json',{'sequences':seqs,'train':tr.tolist(),'validation':va.tolist(),
                         'positive_split_seed':42,'negative_split_seed':4242,'heldout_positive_sequences_shared_between_conditions':True})
    pd.DataFrame(history).to_csv(out/'training.csv',index=False)
    del model,opt;torch.cuda.empty_cache()
    from external_metrics import panel,from_peptides,evaluate
    pep,_=panel();predictor=Predictor(out);yp=predictor.predict(pep.sequence.tolist())
    if not np.isfinite(yp).all():raise ValueError('Nonfinite external predictions')
    series=pd.Series(yp,index=pep.sequence,name='log10mic');series.to_csv(out/'predictions.csv')
    met,fam,edges=evaluate(from_peptides(series,unit='log10_ug_mL'))
    dump(out/'metrics.json',met);fam.to_csv(out/'family_metrics.csv',index=False);edges.to_csv(out/'pair_predictions.csv',index=False)
    dump(out/'DONE.json',{'status':'PASS','smoke':smoke,'elapsed_seconds':time.time()-start,'seed':seed,'condition':condition,
         'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'schema':SCHEMA,'code_sha256':sha(__file__)})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=['with_negatives','without_negatives'],required=True)
    ap.add_argument('--seed',type=int,default=123);ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.prepare:
        d=cached_static(a.condition);print('AMPLiT static features ready',a.condition,len(d[0]),len(d[4]),flush=True)
    else:train(a.condition,a.seed,a.gpu,a.smoke)


if __name__=='__main__':main()
