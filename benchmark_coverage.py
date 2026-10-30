"""Benchmark coverage matrix for Pei et al. and Platonov et al.

This module operates ONLY on compact group summaries. It never accesses W&B,
and never requests or retains individual runs.
"""
from itertools import product
import re

from core import TARGET

# The first entry is intentionally 'nsp' (as specified), not 'nsd'.
MODELS = (
    "nsp",
    "poly_nsd",
    "bsnn",
    "joint_diff",
    "sheaf_deepset",
    "sheaf_set_transformer",
    "sheaf_gin",
    "id_sheaf",
    "conn_nsd",
    "sheaf_att",
    "rot_inv",
)

# Pei et al. 2020 / Geom-GCN: the Actor dataset is often logged as "film".
PEI_DATASETS = (
    "texas", "wisconsin", "cornell", "chameleon", "squirrel", "film",
)

# Five new benchmark datasets introduced by Platonov et al. (ICLR 2023).
PLATONOV_DATASETS = (
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions",
)

# Cleaned/filtered variants discussed in Platonov et al. are optional.
FILTERED_DATASETS = ("chameleon_filtered", "squirrel_filtered")

RESTRICTION_MAPS = ("diagonal", "general", "orthogonal", "low_rank")


def _token(value):
    """Conservative case, whitespace, dash and underscore normalization."""
    value = str(value if value is not None else "").strip().casefold()
    value = re.sub(r"[\s-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_model(value):
    # Do not equate nsp to nsd, or otherwise collapse different architectures.
    return _token(value)


def normalize_dataset(value):
    token = _token(value)
    return {
        "actor": "film",
        "romanempire": "roman_empire",
        "amazonratings": "amazon_ratings",
    }.get(token, token)


def normalize_restriction_map(value):
    token = _token(value)
    return {
        "lowrank": "low_rank",
        "low_rank": "low_rank",
        "low_rank_map": "low_rank",
    }.get(token, token)


def benchmark_datasets(include_filtered=False):
    datasets = (
        tuple((name, "Pei et al. (2020)") for name in PEI_DATASETS)
        + tuple((name, "Platonov et al. (2023)") for name in PLATONOV_DATASETS)
    )
    if include_filtered:
        datasets += tuple(
            (name, "Platonov: filtered variant") for name in FILTERED_DATASETS
        )
    return datasets


def coverage_matrix(groups, include_filtered=False, target=TARGET):
    """Cross expected configs with available, *finished* compact W&B groups.

    Groups are already restricted to creator fs395. Deduplicate source aliases
    (e.g. Actor/Film and low-rank/low_rank) by summing their finished runs.
    Return ALL expected combinations, including completely missing ones.
    """
    counts = {}
    for row in groups:
        if _token(row.get("state", "finished")) != "finished":
            continue
        key = (
            normalize_model(row.get("model", "")),
            normalize_dataset(row.get("dataset", "")),
            normalize_restriction_map(row.get("transport_type", "")),
        )
        count = max(0, int(row.get("finished") or 0))
        counts[key] = counts.get(key, 0) + count

    output = []
    for model, (dataset, benchmark), restriction_map in product(
        MODELS, benchmark_datasets(include_filtered), RESTRICTION_MAPS
    ):
        count = counts.get((model, dataset, restriction_map), 0)
        status = (
            "Missing" if count == 0
            else "Complete" if count >= target
            else "Incomplete"
        )
        output.append({
            "model": model,
            "dataset": dataset,
            "restriction_maps": restriction_map,
            "benchmark": benchmark,
            "finished_runs": count,
            "missing_runs": max(0, target - count),
            "completion_pct": min(100.0, 100.0 * count / target),
            "status": status,
        })
    rank = {"Missing": 0, "Incomplete": 1, "Complete": 2}
    return sorted(output, key=lambda row: (
        rank[row["status"]], row["model"], row["dataset"], row["restriction_maps"]
    ))
