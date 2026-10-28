"""PyG graph construction for the HC50 toxicity task, reusing the TIGER stack.

Builds per-residue CA-contact graphs (``code/dataloader.construct_graph``)
for peptides in a toxin labeled CSV. Unlike the MIC pipeline, the label is the
**binary toxicity label** (1 = toxin, HC50 <= threshold) stored in
``data.gt`` directly — no log2(MIC) transform.

Node features / global features / seq codes are identical to the MIC graphs:
20 AA descriptors + 20 Rosetta per-residue energies (+3 coords if configured),
10-D physicochemical ``p_f`` (raw at cache time; z-scored per fold by the
trainer via ``code/utils/scaling.py``).

Also provides ``build_panel_graphs`` for external panels where some sequences
may lack a PDB: with ``allow_placeholder=True`` a 1-node self-loop graph is
substituted (only sound for modality configs without "g" — the encoder zeroes
the graph representation anyway via ``feature_modalities``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from code.dataloader import PeptideGraph, construct_graph, load_aa_features
from code.utils.config import Config
from code.utils.features import tabular_features


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"toxin_graph_cache_{key}.pt"


def toxin_cache_key(label_csv: str | Path, cfg: Config) -> str:
    stem = Path(label_csv).stem
    coords = int(bool(getattr(cfg, "include_node_coords", False)))
    return f"{stem}_len{cfg.max_len}_coords{coords}"


def _prepare_rows(df: pd.DataFrame, pdb_dir: str | Path) -> pd.DataFrame:
    work = df.copy()
    for col, default in (("n_terminus", ""), ("c_terminus", "")):
        if col not in work.columns:
            work[col] = default
    if "cfu_group" not in work.columns:
        work["cfu_group"] = "unknown"
    work["sequence"] = work["sequence"].astype(str).str.upper().str.strip()
    work["label"] = work["label"].astype(int)
    tabs = tabular_features(work[["sequence", "n_terminus", "c_terminus"]], include_cfu=False, property_scaling="raw")
    work["tab_f"] = [row for row in tabs]
    work["pdb_path"] = [str(Path(pdb_dir) / f"{s}.pdb") for s in work["sequence"]]
    return work


def build_toxin_graphs(
    label_csv: str | Path,
    pdb_dir: str | Path,
    cfg: Config,
    *,
    cache_dir: str | Path = "artifacts/cache_toxin",
    force_rebuild: bool = False,
) -> tuple[list[PeptideGraph], pd.DataFrame, dict[str, Any]]:
    """Graphs + kept rows for a toxin labeled CSV (requires PDB for every seq)."""
    cache_file = _cache_path(cache_dir, toxin_cache_key(label_csv, cfg))
    if cache_file.exists() and not force_rebuild:
        payload = torch.load(cache_file, weights_only=False)
        return payload["graphs"], payload["kept_df"], payload["stats"]

    df = pd.read_csv(label_csv)
    work = _prepare_rows(df, pdb_dir)
    aa_features = load_aa_features(cfg.feature_path)
    graphs, kept, failed = [], [], 0
    for row in work.itertuples():
        graph = construct_graph(row, aa_features, cfg)
        if graph is None:
            failed += 1
            continue
        graphs.append(graph)
        kept.append(row._asdict())
    stats = {
        "input_rows": int(len(work)),
        "graph_rows": int(len(graphs)),
        "graph_failed_rows": int(failed),
        "pdb_dir": str(pdb_dir),
        "label_csv": str(label_csv),
    }
    kept_df = pd.DataFrame(kept)
    torch.save({"graphs": graphs, "kept_df": kept_df, "stats": stats}, cache_file)
    return graphs, kept_df, stats


def _placeholder_graph(sequence: str, aa_features: dict[str, list[float]], cfg: Config) -> PeptideGraph:
    """1-node self-loop graph used when no PDB exists and the graph modality is off."""
    from code.utils.constants import encode_sequence
    from code.utils.features import calculate_property

    aa = sequence[0] if sequence[0] in aa_features else "A"
    x = torch.tensor([aa_features[aa] + [0.0] * 20], dtype=torch.float32)
    edge = torch.tensor([[0], [0]], dtype=torch.long)
    g = PeptideGraph(edge, x)
    g.seq = torch.tensor([encode_sequence(sequence, cfg.max_len)], dtype=torch.float32)
    g.p_f = torch.tensor([calculate_property(sequence, "", "", scaling="raw")], dtype=torch.float32)
    g.tab_f = torch.zeros((1, 10), dtype=torch.float32)
    g.cfu_id = torch.zeros((1,), dtype=torch.long)
    g.gt = 0.0
    g.sequence = sequence
    return g


def build_panel_graphs(
    df: pd.DataFrame,
    pdb_dirs: list[str | Path],
    cfg: Config,
    *,
    allow_placeholder: bool = False,
) -> tuple[list[PeptideGraph], pd.DataFrame, dict[str, Any]]:
    """Graphs for an external panel; PDBs resolved across ``pdb_dirs`` in order.

    When none of the dirs has ``<SEQ>.pdb``: fail the row, or substitute a
    1-node placeholder graph when ``allow_placeholder`` (only sound when the
    graph modality "g" is disabled — its representation is zeroed anyway).
    """
    work = _prepare_rows(df, "unused")
    aa_features = load_aa_features(cfg.feature_path)
    pools = [{p.stem for p in Path(d).glob("*.pdb")} for d in pdb_dirs]
    graphs, kept, failed, placeholders = [], [], 0, 0
    for row in work.itertuples():
        pdb_path = None
        for pool, base in zip(pools, pdb_dirs):
            if row.sequence in pool:
                pdb_path = str(Path(base) / f"{row.sequence}.pdb")
                break
        if pdb_path is None:
            if not allow_placeholder:
                failed += 1
                continue
            graph = _placeholder_graph(row.sequence, aa_features, cfg)
            placeholders += 1
        else:
            patched = row._replace(pdb_path=pdb_path)
            graph = construct_graph(patched, aa_features, cfg)
            if graph is None:
                if not allow_placeholder:
                    failed += 1
                    continue
                graph = _placeholder_graph(row.sequence, aa_features, cfg)
                placeholders += 1
        graphs.append(graph)
        kept.append(row._asdict())
    stats = {
        "input_rows": int(len(work)),
        "graph_rows": int(len(graphs)),
        "graph_failed_rows": int(failed),
        "placeholder_graphs": int(placeholders),
        "pdb_dirs": [str(d) for d in pdb_dirs],
    }
    return graphs, pd.DataFrame(kept), stats
