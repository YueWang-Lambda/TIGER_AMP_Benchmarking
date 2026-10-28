"""Register the full approved experiment grid before viewing any new scores."""
from copy import deepcopy
from common import *
from code.utils.config import load_config

NO_P={'s4_mod_sg','s4_mod_g','s4_mod_s'}


def effective(c):
    c=deepcopy(c)
    for k in ['name','out_dir','device','shared_cache_dir','checkpoint','smoke']:c.pop(k,None)
    if c['model_kind']=='single' or not c['neg_train_inject']:
        for k in ['neg_pair_mode','neg_pair_gate','neg_pair_topk','neg_max_pairs']:c.pop(k,None)
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def main():
    assert (WORK/'freeze_complete.json').exists()
    config_dir=WORK/'configs';config_dir.mkdir(parents=True,exist_ok=True)
    rows=[];lookup={}
    def register(label,seed,c,group,reuse=None):
        c.update(seed=seed,validation_similarity_threshold=.7,train_similarity_gate_enabled=c.get('train_similarity_gate_enabled',True),
                 shared_cache_dir=str(WORK/'graph_cache'),device='cuda:0',smoke=False)
        key=effective(c)
        if key in lookup:
            rows.append({'label':label,'seed':seed,'group':group,'action':'alias','canonical_task':lookup[key], 'config_hash':key});return
        task=f'{label}_seed{seed}';lookup[key]=task
        c.update(name=task,out_dir=str(RUNS/label/task))
        dest=config_dir/f'{task}.json';dump(dest,c)
        row={'label':label,'seed':seed,'group':group,'task_id':task,'config_hash':key,
             'config_path':str(dest),'run_dir':str(RUNS/label/task),'action':'reuse_validate' if reuse else 'train'}
        if reuse:row['original_run_dir']=str(reuse)
        rows.append(row)
    for arm in arms():
        for seed in range(120,130):
            cp=run_config(arm,seed);c=load_config(cp).to_dict()
            register(arm,seed,c,'ablation40',cp.parent if arm in NO_P else None)
    for seed in range(120,130):
        base=load_config(run_config('S4',seed)).to_dict()
        for kind in ['pair','single']:
            for n2,n1 in [(0,0),(1,0),(0,1),(1,1)]:
                c=deepcopy(base);c.update(model_kind=kind,cls_pretrain_epochs=15 if n2 else 0,
                    neg_train_inject=bool(n1),neg_pair_mode='joint',single_pretrain_epochs=0,
                    anchor_multitask=0.0,anchor_neg_rows=False,
                    neg_csv=base['neg_csv'] if n2 or n1 else None)
                label=f'factor_{"P" if kind=="pair" else "S"}{n2}{n1}'
                register(label,seed,c,'negative_factorial8')
        c=deepcopy(base);c['train_similarity_gate_enabled']=False
        register('s4_gate_off',seed,c,'gate_off')
    assert sum(r['group']=='ablation40' for r in rows)==400
    assert sum(r['group']=='negative_factorial8' for r in rows)==80
    assert sum(r['group']=='gate_off' for r in rows)==10
    dump(WORK/'run_registry.json',rows)
    counts={a:sum(r['action']==a for r in rows) for a in ['train','reuse_validate','alias']}
    dump(WORK/'registry_summary.json',{'counts':counts,'total_result_slots':len(rows),'seeds':list(range(120,130)),
                                      'ablation_arms':40,'factorial_configs':8,'gate_off_configs':1,
                                      'amplit_separate_training_tasks':20})
    print(counts,flush=True)


if __name__=='__main__':main()
