"""Strict, identity-aligned metrics on the unchanged canonical external endpoint."""
import numpy as np
import pandas as pd
from common import *
from code.utils.metrics import regression_metrics

KEYS=['family','anchor_sequence','query_sequence']
METRICS=['PCC','KCC','log10MAE','RSE']


def panel():
    pep=pd.read_csv(PKG/'2_external_testset/data/final11_peptides.csv')
    edges=pd.read_csv(PKG/'4_S4_benchmark/results/pair_keys_primary_2211.csv')
    pep.family=pep.family.str.rstrip('/');edges.family=edges.family.str.rstrip('/')
    assert len(pep)==208 and pep.sequence.is_unique and len(edges)==2211 and edges.family.nunique()==11
    assert not edges.duplicated(KEYS).any()
    assert set(edges.anchor_sequence)|set(edges.query_sequence)==set(pep.sequence)
    mic=pep.set_index('sequence').MIC_E_coli_ug_mL
    edges['y_true']=np.log2(edges.anchor_sequence.map(mic)/edges.query_sequence.map(mic))
    assert np.isfinite(edges.y_true).all()
    return pep,edges


def from_peptides(pred, *, unit):
    pep,edges=panel()
    if not isinstance(pred,pd.Series) or not pred.index.is_unique or set(pred.index)!=set(pep.sequence):
        raise ValueError('Prediction peptide identity set must be exactly 208 unique endpoints')
    if not np.isfinite(pred.to_numpy(float)).all(): raise ValueError('Nonfinite peptide prediction')
    if unit not in ['log10_ug_mL','log2_ug_mL']: raise ValueError('Explicit mass-concentration log unit required')
    values=(edges.anchor_sequence.map(pred)-edges.query_sequence.map(pred)).to_numpy(float)
    if unit=='log10_ug_mL':values/=np.log10(2)
    edges['y_pred']=values
    return edges


def evaluate(frame):
    _,canonical=panel()
    data=frame.copy()
    if not set(KEYS+['y_pred']).issubset(data.columns):raise ValueError('Missing prediction fields')
    data.family=data.family.str.rstrip('/')
    if data.duplicated(KEYS).any():raise ValueError('Duplicate pair identity')
    if set(map(tuple,data[KEYS].to_numpy()))!=set(map(tuple,canonical[KEYS].to_numpy())):
        raise ValueError('Missing or extra canonical pair identities')
    ordered=canonical.merge(data[KEYS+['y_pred']],on=KEYS,how='left',validate='one_to_one')
    if not np.isfinite(ordered.y_pred.to_numpy(float)).all():raise ValueError('Nonfinite pair prediction')
    if 'y_true' in data:
        ref=canonical.merge(data[KEYS+['y_true']],on=KEYS,validate='one_to_one',suffixes=('_ref','_provided'))
        if not np.allclose(ref.y_true_ref,ref.y_true_provided,rtol=0,atol=1e-7):raise ValueError('Truth/unit mismatch')
    per=[]
    for fam,g in ordered.groupby('family',sort=True):
        met=regression_metrics(g.y_true,g.y_pred)
        per.append({'family':fam,'n_pairs':len(g),**{k:met[k] for k in METRICS}})
    per=pd.DataFrame(per)
    # No nanmean: undefined family correlations remain undefined for the macro.
    macro={k:(float(per[k].mean()) if np.isfinite(per[k]).all() else None) for k in METRICS}
    undefined={k:per.loc[~np.isfinite(per[k]),'family'].tolist() for k in METRICS}
    sensitivity={k:float(per[k].fillna(0).mean()) for k in ['PCC','KCC']}
    return {'macro':macro,'undefined_families':undefined,'zero_contribution_sensitivity':sensitivity,
            'n_pairs':2211,'n_families':11},per,ordered
