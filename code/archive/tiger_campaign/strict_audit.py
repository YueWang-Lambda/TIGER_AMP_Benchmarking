"""Observe the actual strict validation table without changing scientific execution."""
from common import *

def install_recording(module, run):
    original = module.build_within_fold_strict_pairs
    calls = []
    def recording(rows, cfg, cap, seed):
        table = original(rows, cfg, cap, seed)
        fold = len(calls) + 1
        assert seed == cfg.seed + fold and fold <= 5
        assert len(table) > 0 and len(table) <= cap
        keys = ['anchor_sequence', 'query_sequence']
        allowed = set(rows.sequence)
        assert set(table.anchor_sequence) | set(table.query_sequence) <= allowed
        # Frozen builder excludes identical ROW indices, not identical sequences.
        # Repeated sequence records therefore retain repeated pair occurrences.
        # Preserve their exact multiplicities rather than silently deduplicating.
        target = run / 'intermediate' / f'fold{fold}_strict_validation_pairs.csv'
        table.to_csv(target, index=False)
        calls.append(fold)
        dump(target.with_suffix('.json'), {'fold': fold, 'sampling_seed': seed,
             'validation_rows': len(rows), 'n_pairs': len(table), 'cap': cap,
             'sha256': sha(target), 'both_endpoints_in_validation_fold': True,
             'duplicate_sequence_pair_occurrences': int(table.duplicated(keys).sum()),
             'same_sequence_endpoint_occurrences': int((table.anchor_sequence==table.query_sequence).sum()),
             'source': 'captured actual table passed to checkpoint-selection evaluation'})
        return table
    module.build_within_fold_strict_pairs = recording

