"""W&B finished-run monitoring and validation-based model selection.

Every finished run from creator fs395 is eligible, regardless of model family.
The *best* run per group is selected solely by the appropriate validation metric.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

PROJECT = "adavit/sheaf-mpnn-runs"
TARGET = 250
CREATOR = "fs395"
KEYS = ("state", "model", "dataset", "transport_type")
BASE_URL = "https://api.forge.coreweave.com"
ROC_DATASETS = frozenset({"minesweeper", "tolokers", "questions"})
TEST_COLUMNS = (
    "mean_test_acc",
    "std_test_acc",
    "mean_test_roc_auc",
    "std_test_roc_auc",
)


def _user_name(run):
    """Read the actual run creator, not the API key owner or project entity."""
    user = (getattr(run, "_attrs", {}) or {}).get("user")
    if not user:
        user = getattr(run, "user", None)
    if isinstance(user, dict):
        return user.get("username")
    return getattr(user, "username", None)


def _plain(value):
    # W&B sometimes wraps values as {"value": ..., "_type": ...}.
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None or str(value).strip() == "":
        return "(missing)"
    return str(value)


def _config_str(config, key):
    return _plain((config or {}).get(key, "(missing)"))


def _model_name(config):
    """Keep the W&B model field, with fallbacks for other training pipelines."""
    config = config or {}
    for key in ("model", "model_name", "model_type", "architecture", "arch", "network", "model.name", "model.type"):
        value = config.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            value = value.get("value", value)
            if isinstance(value, dict):
                for nested in ("name", "model_name", "type", "_target_", "class_path"):
                    if value.get(nested):
                        value = value[nested]
                        break
        result = _plain(value)
        if result != "(missing)":
            return result
    return "(missing)"


def _number(value):
    """A missing, nonfinite or nonnumeric metric must never win a ranking."""
    try:
        if isinstance(value, bool):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _summary_value(summary, name):
    """Read exact W&B run summary keys (including JSON-encoded summaries)."""
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except (TypeError, ValueError):
            return None
    try:
        return _number(summary.get(name))
    except (AttributeError, TypeError, ValueError):
        return None


def metric_for(dataset):
    """Choose ROC-AUC on the three specified datasets; accuracy otherwise."""
    if str(dataset).strip().casefold() in ROC_DATASETS:
        return "mean_val_roc_auc", "mean_test_roc_auc", "std_test_roc_auc"
    return "mean_val_acc", "mean_test_acc", "std_test_acc"


def invalid_metric_fields(row):
    """Exact required fields missing/nonfinite for a single run."""
    return tuple(
        name for name in metric_for(row["dataset"])
        if _number(row.get(name)) is None
    )


def _metric_key_names(summary):
    """Metadata only: metric-like summary key names to help diagnose missing scores."""
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except (TypeError, ValueError):
            return []
    if not hasattr(summary, "keys"):
        return []
    return sorted(
        key for key in summary.keys()
        if isinstance(key, str) and any(
            token in key.casefold() for token in ("acc", "auc", "roc", "test", "val")
        )
    )[:80]


def fetch(progress=None):
    """Read finished runs in pages and validate the creator client-side too."""
    import wandb

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY is missing from Streamlit secrets")
    api = wandb.Api(
        api_key=key,
        overrides={"base_url": os.getenv("WANDB_BASE_URL", BASE_URL)},
        timeout=45,
    )
    # Restrict user and state, but deliberately DO NOT restrict the model.
    runs = api.runs(
        PROJECT,
        filters={"state": "finished", "username": CREATOR},
        per_page=100,
        lazy=False,
    )
    rows = []
    examined = unknown = 0
    for run in runs:
        examined += 1
        username = _user_name(run)
        if not username:
            unknown += 1
            continue
        if username.casefold() != CREATOR or str(run.state).lower() != "finished":
            continue

        config = run.config or {}
        summary = run.summary or {}
        rows.append({
            "state": "finished",
            "model": _model_name(config),
            "dataset": _config_str(config, "dataset"),
            "transport_type": _config_str(config, "transport_type"),
            "run_id": str(run.id),
            "name": str(run.name or run.id),
            "url": str(getattr(run, "url", "") or ""),
            "available_metric_keys": _metric_key_names(summary),
            **{
                name: _summary_value(summary, name)
                for name in (
                    "mean_val_acc",
                    "mean_test_acc",
                    "std_test_acc",
                    "mean_val_roc_auc",
                    "mean_test_roc_auc",
                    "std_test_roc_auc",
                )
            },
        })
        if progress is not None and examined % 50 == 0:
            progress(examined, len(rows))
    if progress is not None:
        progress(examined, len(rows))
    return {"runs": rows, "examined": examined, "unknown": unknown}


def aggregate(runs):
    """For each (state,model,dataset,transport_type), choose max validation.

    The displayed mean/std test values BOTH come from the SAME winning run.
    Do not use test metrics to select a run. Missing validation yields N/A.
    """
    counts = Counter()
    winners = {}
    validation_counts = Counter()
    complete_metric_counts = Counter()
    possible_keys = {}
    for run in runs:
        group = tuple(run[k] for k in KEYS)
        counts[group] += 1
        possible_keys.setdefault(group, set()).update(run.get("available_metric_keys", []))
        if not invalid_metric_fields(run):
            complete_metric_counts[group] += 1
        validation_name, _, _ = metric_for(run["dataset"])
        val = _number(run.get(validation_name))
        if val is None:
            continue
        validation_counts[group] += 1
        previous = winners.get(group)
        if previous is None or val > previous[0] or (
            val == previous[0] and str(run.get("run_id", "")) < str(previous[1].get("run_id", ""))
        ):
            winners[group] = (val, run)

    result = []
    for group, count in counts.items():
        val_name, mean_name, std_name = metric_for(group[KEYS.index("dataset")])
        best = winners.get(group)
        chosen = best[1] if best else None
        row = {
            **dict(zip(KEYS, group)),
            "finished": count,
            "remaining": max(0, TARGET - count),
            "progress": min(100.0, 100.0 * count / TARGET),
            "status": ("Complete" if count >= TARGET
                       else "Almost there" if count >= 200 else "In progress"),
            "selection_metric": val_name,
            "valid_validation_runs": validation_counts[group],
            "valid_all_metrics_runs": complete_metric_counts[group],
            "invalid_metric_runs": count - complete_metric_counts[group],
            "available_metric_keys": sorted(possible_keys.get(group, set())),
            "best_validation": best[0] if best else None,
            "best_run_id": chosen.get("run_id") if chosen else None,
            "best_run_name": chosen.get("name") if chosen else None,
            "best_run_url": chosen.get("url") if chosen else None,
            **{key: None for key in TEST_COLUMNS},
        }
        if chosen is not None:
            row[mean_name] = _number(chosen.get(mean_name))
            row[std_name] = _number(chosen.get(std_name))
        result.append(row)

    return sorted(
        result,
        key=lambda row: (
            -row["finished"],
            row["model"].casefold(),
            row["dataset"].casefold(),
            row["transport_type"].casefold(),
        ),
    )


def group_id(row):
    return json.dumps([row[k] for k in KEYS], separators=(",", ":"))
