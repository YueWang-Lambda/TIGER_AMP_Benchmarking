"""Check actual training products against the registered U20000 experiment."""
from common import *


def check_training_identity(run, registration):
    cfg = load(run/'config.json')
    expected = load(registration['config_path'])
    assert cfg == expected, ('Runtime configuration differs', run)
    reference = Path(registration['sampling_reference_run'])
    base = load(reference/'config.json')
    sampling_fields = ['pair_balance_num','use_signed_sampling','similarity_threshold','train_similarity_gate_enabled']
    same_sampling = all(cfg[k] == base[k] for k in sampling_fields)
    checks = []
    for fold in range(1, 6):
        for name in [f'fold{fold}_indices.json', f'p_feature_zscore_fold{fold}.json']:
            assert load(run/'intermediate'/name) == load(reference/'intermediate'/name), (run,name)
        audit = load(run/f'intermediate/fold{fold}_pair_audit.json')
        assert audit['positive_pairs'] > 0
        if same_sampling:
            old = load(reference/f'intermediate/fold{fold}_pair_audit.json')
            for key in ['positive_pairs','positive_pair_sha256']:
                assert audit[key] == old[key], (run,fold,key)
        if registration['label'] == 'factor_P00':
            assert audit['negative_pairs'] == 0 and cfg['cls_pretrain_epochs'] == 0 and not cfg['neg_train_inject']
            assert not (run/f'logs/fold{fold}_clspretrain.csv').exists()
        else:
            assert cfg['cls_pretrain_epochs'] == 15 and cfg['neg_train_inject']
            assert (run/f'logs/fold{fold}_clspretrain.csv').exists()
        if cfg['val_pair_mode'] == 'within_fold_strict':
            actual = run/f'intermediate/fold{fold}_strict_validation_pairs.csv'
            expected_table = WORK/f'expected_validation/seed{cfg["seed"]}_fold{fold}.csv'
            assert sha(actual) == sha(expected_table), (run,fold,'strict table')
        checks.append({'fold':fold,'split_and_global_scaler_match':True,
                       'positive_pair_identity_matches_reference':same_sampling,
                       'positive_pairs':audit['positive_pairs'],'negative_pairs':audit['negative_pairs']})
    identity = load(run/'intermediate/input_identity.json')
    prior = load(reference/'intermediate/input_identity.json')
    for key in ['n_training_rows','n_sequences','ACT_rows','cpu_threads']:
        assert identity[key] == prior[key], (run,key)
    return {'status':'PASS','checks':checks,'input_identity_matches':True,
            'reference_run':str(reference),'configuration_matches_registry':True}
