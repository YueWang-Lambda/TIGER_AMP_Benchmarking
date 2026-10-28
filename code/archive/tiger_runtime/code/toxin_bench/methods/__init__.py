"""Benchmark method adapters. Each exposes ``adapter(seed, gpu, fold) ->
object with .fit(df_train) / .predict_proba(df)``; hyperparameters follow the
respective paper/README and are recorded in each module's ``META``."""

from __future__ import annotations

from code.toxin_bench.methods import hemopi1, hemopi2, hemopred, lysepred, mlpeptide, plisson

METHODS: dict[str, type] = {
    "hemopi1": hemopi1.adapter,
    "hemopi2": hemopi2.adapter,
    "hemopred": hemopred.adapter,
    "mlpeptide_rf": mlpeptide.rf_adapter,
    "mlpeptide_rnn": mlpeptide.rnn_adapter,
    "lysepred": lysepred.adapter,
    "plisson": plisson.adapter,
}

_META: dict[str, dict] = {
    "hemopi1": hemopi1.META,
    "hemopi2": hemopi2.META,
    "hemopred": hemopred.META,
    "mlpeptide_rf": mlpeptide.META_RF,
    "mlpeptide_rnn": mlpeptide.META_RNN,
    "lysepred": lysepred.META,
    "plisson": plisson.META,
}

try:  # AMPDeep pulls in transformers/ProtBERT — optional at import time
    from code.toxin_bench.methods import ampdeep

    METHODS["ampdeep"] = ampdeep.adapter
    _META["ampdeep"] = ampdeep.META
except Exception as _exc:  # pragma: no cover
    import warnings

    warnings.warn(f"ampdeep adapter unavailable: {_exc}")

try:  # HemoNet needs the precomputed SeqVec embeddings
    from code.toxin_bench.methods import hemonet

    METHODS["hemonet"] = hemonet.adapter
    _META["hemonet"] = hemonet.META
except Exception as _exc:  # pragma: no cover
    import warnings

    warnings.warn(f"hemonet adapter unavailable: {_exc}")

try:  # FusionPeptide sequence-encoder variants (rnn/lstm/transformer/mamba)
    from code.toxin_bench.methods import fusionpeptide_variants as fpv

    for _kind in ("rnn", "lstm", "transformer", "mamba"):
        METHODS[f"fp_{_kind}"] = fpv.adapter_factory(_kind)
        _META[f"fp_{_kind}"] = fpv.META[f"fp_{_kind}"]
except Exception as _exc:  # pragma: no cover
    import warnings

    warnings.warn(f"fusionpeptide variants unavailable: {_exc}")


def method_meta(name: str) -> dict:
    return _META.get(name, {})
