"""Build corrected raw graph caches once, before concurrent GPU workers."""
import argparse
from common import *
import torch
torch.set_num_threads(2)
from code.utils.config import load_config
from code.dataloader import load_or_build_cache,load_or_build_neg_cache

ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
c=load_config(a.config)
print('Building corrected positive cache',a.config,flush=True)
p=load_or_build_cache(c,ROOT)
assert len(p['rows'])==7217 and len(p['graphs'])==7217
assert p['rows'].n_terminus.fillna('').eq('ACT').sum()==209
print('Positive cache PASS',p['train_stats'],flush=True)
n=load_or_build_neg_cache(c,ROOT)
assert len(n['neg_rows'])==7860
print('Negative cache PASS',n['neg_stats'],flush=True)
