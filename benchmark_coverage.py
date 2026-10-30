"""Repository-grounded Pei/Platonov expected sweep coverage.

Model-specific variants are from:
alessioborgi/pytorch-SheafNeuralNetworks-working
src/exp/registries/presets.py and configs/<dataset>/<model>/*.yaml.

This operates only on compact finished-run summaries; no W&B API calls.
"""
from itertools import product
import re

from core import TARGET

RESTRICTION_MAPS = ("diagonal", "general", "orthogonal", "low_rank")

# The 8 model families whose expected sweep map choices can be verified in
# the source repository. It is incorrect to cross every family with four maps.
MODEL_VARIANTS = {
    "nsp": RESTRICTION_MAPS,
    "poly_nsd": RESTRICTION_MAPS,
    "bsnn": RESTRICTION_MAPS,
    "joint_diff": ("general",),
    "id_sheaf": ("diagonal",),   # identity transport, logged diagonal
    "conn_nsd": ("orthogonal",), # feature-derived orthogonal transport
    "sheaf_attn": ("res_sheafan",), # only shipped sweep YAML
    "rot_inv": ("general",),
}

# These were requested but are not in the inspected repo's registries or
# sweep YAMLs. Do not label invented configurations as "Missing".
UNVERIFIED_MODELS = (
    "sheaf_deepset", "sheaf_set_transformer", "sheaf_gin",
)
MODELS = tuple(MODEL_VARIANTS)
ALL_REQUESTED_MODELS = MODELS + UNVERIFIED_MODELS

PEI_DATASETS = (
    "texas", "wisconsin", "cornell", "chameleon", "squirrel", "film",
)
PLATONOV_DATASETS = (
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions",
)
FILTERED_DATASETS = ("chameleon_filtered", "squirrel_filtered")


def _token(value):
    value = str(value if value is not None else "").strip().casefold()
    value = re.sub(r"[\s-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_model(value):
    token = _token(value)
    # The user wrote "sheaf_att"; the training registry logs "sheaf_attn".
    return {"sheaf_att": "sheaf_attn", "sheafan": "sheaf_attn"}.get(token, token)


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
        "low_rank_map": "low_rank",
        "res_sheaf_an": "res_sheafan",
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


def _counts(groups):
    counts = {}
    for row in groups:
        if _token(row.get("state", "finished")) != "finished":
            continue
        key = (
            normalize_model(row.get("model")),
            normalize_dataset(row.get("dataset")),
            normalize_restriction_map(row.get("transport_type")),
        )
        count = max(0, int(row.get("finished") or 0))
        counts[key] = counts.get(key, 0) + count
    return counts


def coverage_matrix(groups, include_filtered=False, target=TARGET):
    """Generate only verified expected model/dataset/variant combinations."""
    counts = _counts(groups)
    output = []
    for model, variants in MODEL_VARIANTS.items():
        for (dataset, benchmark), variant in product(
            benchmark_datasets(include_filtered), variants
        ):
            count = counts.get((model, dataset, variant), 0)
            status = (
                "Missing" if count == 0
                else "Complete" if count >= target
                else "Incomplete"
            )
            output.append({
                "model": model,
                "dataset": dataset,
                "restriction_maps": variant,
                "benchmark": benchmark,
                "finished_runs": count,
                "missing_runs": max(0, target - count),
                "completion_pct": min(100.0, 100.0 * count / target),
                "status": status,
            })
    rank = {"Missing": 0, "Incomplete": 1, "Complete": 2}
    return sorted(
        output,
        key=lambda row: (
            rank[row["status"]], row["model"],
            row["dataset"], row["restriction_maps"],
        ),
    )


def unverified_observations(groups, include_filtered=False):
    """Show observed W&B variants for unverified model families, without
    claiming that absent variants constitute missing experiments."""
    datasets = {name for name, _ in benchmark_datasets(include_filtered)}
    output = []
    for (model, dataset, variant), count in sorted(_counts(groups).items()):
        if model in UNVERIFIED_MODELS and dataset in datasets:
            output.append({
                "model": model, "dataset": dataset,
                "transport_type_observed": variant,
                "finished_runs": count,
                "verification": "Observed in W&B; model map grid unverified",
            })
    return output
