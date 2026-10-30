"""Memory-bounded W&B run monitoring.

Fetch one small GraphQL page at a time and retain only per-group statistics,
the best run, and a bounded sample for diagnostics. Unlike wandb.Api().runs(),
this does not retain all previously visited Run objects and their summaries.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

PROJECT = "adavit/sheaf-mpnn-runs"
CREATOR = "fs395"
TARGET = 250
KEYS = ("state", "model", "dataset", "transport_type")
ROC_DATASETS = frozenset({"minesweeper", "tolokers", "questions"})
TEST_COLUMNS = ("mean_test_acc", "std_test_acc", "mean_test_roc_auc", "std_test_roc_auc")
BASE_URL = "https://api.forge.coreweave.com"
RUN_BASE_URL = "https://forge.coreweave.com/wandb/adavit/sheaf-mpnn-runs/runs/"
SAMPLE_SIZE = 5

# Unlike the full W&B RunFragment, this intentionally omits systemMetrics,
# historyKeys, tags, descriptions, and other unrelated fields.
_RUNS_QUERY = """
query CompactRunMonitor(
  $entity: String!, $project: String!, $cursor: String,
  $perPage: Int!, $filters: JSONString
) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, after: $cursor, first: $perPage) {
      edges {
        cursor
        node {
          name
          displayName
          state
          user { username }
          config
          summaryMetrics
        }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
"""


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _plain(value):
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None or str(value).strip() == "":
        return "(missing)"
    return str(value)


def _model_name(config):
    for key in (
        "model", "model_name", "model_type", "architecture", "arch",
        "network", "model.name", "model.type",
    ):
        value = config.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            value = value.get("value", value)
            if isinstance(value, dict):
                for subkey in ("name", "model_name", "type", "_target_", "class_path"):
                    if value.get(subkey):
                        value = value[subkey]
                        break
        output = _plain(value)
        if output != "(missing)":
            return output
    return "(missing)"


def _number(value):
    try:
        if isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def metric_for(dataset):
    if str(dataset).strip().casefold() in ROC_DATASETS:
        return "mean_val_roc_auc", "mean_test_roc_auc", "std_test_roc_auc"
    return "mean_val_acc", "mean_test_acc", "std_test_acc"


def invalid_metric_fields(run):
    return tuple(
        name for name in metric_for(run["dataset"])
        if _number(run.get(name)) is None
    )


def _make_group(run):
    selection_metric, _, _ = metric_for(run["dataset"])
    return {
        **{key: run[key] for key in KEYS},
        "finished": 0,
        "valid_validation_runs": 0,
        "valid_all_metrics_runs": 0,
        "best_validation": None,
        "best_run_id": None,
        "best_run_name": None,
        "best_run_url": None,
        "selection_metric": selection_metric,
        **{key: None for key in TEST_COLUMNS},
        # Keep only a bounded number of examples. Never retain all runs.
        "invalid_samples": [],
        "sample_runs": [],
        "available_metric_keys": set(),
    }


def compact_from_pages(pages):
    """Pure, testable page reducer. Each page is immediately discarded."""
    groups = {}
    examined = 0
    matched = 0
    unknown = 0
    for nodes in pages:
        for node in nodes:
            examined += 1
            user = node.get("user") or {}
            username = user.get("username") if isinstance(user, dict) else None
            if not username:
                unknown += 1
                continue
            if username.casefold() != CREATOR or str(node.get("state", "")).casefold() != "finished":
                continue
            matched += 1
            config = _as_dict(node.get("config"))
            summary = _as_dict(node.get("summaryMetrics"))
            run_id = str(node.get("name") or "")
            row = {
                "state": "finished",
                "model": _model_name(config),
                "dataset": _plain(config.get("dataset")),
                "transport_type": _plain(config.get("transport_type")),
                "run_id": run_id,
                "name": str(node.get("displayName") or run_id),
                "url": RUN_BASE_URL + run_id if run_id else "",
                **{
                    name: _number(summary.get(name))
                    for name in (
                        "mean_val_acc", "mean_test_acc", "std_test_acc",
                        "mean_val_roc_auc", "mean_test_roc_auc", "std_test_roc_auc",
                    )
                },
            }
            group_key = tuple(row[key] for key in KEYS)
            if group_key not in groups:
                groups[group_key] = _make_group(row)
            group = groups[group_key]
            group["finished"] += 1

            # Preserve only a few example run IDs for user inspection.
            sample = {
                "run_id": run_id, "name": row["name"], "url": row["url"],
            }
            if len(group["sample_runs"]) < SAMPLE_SIZE:
                group["sample_runs"].append(sample)

            invalid = invalid_metric_fields(row)
            if not invalid:
                group["valid_all_metrics_runs"] += 1
            else:
                if len(group["invalid_samples"]) < SAMPLE_SIZE:
                    group["invalid_samples"].append({
                        **sample, "missing_or_invalid": ", ".join(invalid),
                    })
                if len(group["available_metric_keys"]) < 60:
                    group["available_metric_keys"].update(
                        str(key) for key in summary
                        if isinstance(key, str)
                        and any(token in key.lower() for token in ("acc", "auc", "roc", "test", "val"))
                    )

            validation = row[group["selection_metric"]]
            if validation is not None:
                group["valid_validation_runs"] += 1
                prior = group["best_validation"]
                if (prior is None or validation > prior
                    or (validation == prior and run_id < (group["best_run_id"] or ""))):
                    group["best_validation"] = validation
                    group["best_run_id"] = run_id
                    group["best_run_name"] = row["name"]
                    group["best_run_url"] = row["url"]
                    _, mean, std = metric_for(row["dataset"])
                    group[mean] = row[mean]
                    group[std] = row[std]

    result = []
    for group in groups.values():
        count = group["finished"]
        group["remaining"] = max(0, TARGET - count)
        group["progress"] = min(100.0, 100 * count / TARGET)
        group["status"] = (
            "Complete" if count >= TARGET
            else "Almost there" if count >= 200 else "In progress"
        )
        group["invalid_metric_runs"] = count - group["valid_all_metrics_runs"]
        group["available_metric_keys"] = sorted(group["available_metric_keys"])[:60]
        result.append(group)
    result.sort(key=lambda group: (
        -group["finished"], group["model"], group["dataset"], group["transport_type"]
    ))
    return {
        "groups": result, "examined": examined, "matched": matched,
        "unknown": unknown, "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _graphql_pages(api, page_size=30):
    """Fetch one small W&B GraphQL page at a time, without a growing paginator."""
    entity, project = PROJECT.split("/", 1)
    variables = {
        "entity": entity,
        "project": project,
        "cursor": None,
        "perPage": page_size,
        "filters": json.dumps({"state": "finished", "username": CREATOR}),
    }
    prior_cursor = None
    while True:
        response = api._service_api.execute_graphql(_RUNS_QUERY, variables=variables)
        project_data = response.get("project") if isinstance(response, dict) else None
        if project_data is None:
            raise RuntimeError("W&B project could not be found or accessed.")
        runs_data = project_data.get("runs") or {}
        edges = runs_data.get("edges") or []
        nodes = [edge["node"] for edge in edges if isinstance(edge, dict) and edge.get("node")]
        yield nodes

        page_info = runs_data.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor") or (edges[-1].get("cursor") if edges else None)
        if not cursor or cursor == prior_cursor:
            raise RuntimeError("W&B pagination returned a repeated or missing cursor.")
        prior_cursor = cursor
        variables["cursor"] = cursor
        # Drop any bulky references from the previous response.
        del response, project_data, runs_data, edges, nodes


def fetch():
    """Memory-bounded snapshot. App and hourly checker should call ONLY this."""
    import wandb

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY is missing from Streamlit secrets")
    api = wandb.Api(
        api_key=key,
        overrides={"base_url": os.getenv("WANDB_BASE_URL", BASE_URL)},
        timeout=45,
    )
    return compact_from_pages(_graphql_pages(api))


def group_id(row):
    return json.dumps([row[key] for key in KEYS], separators=(",", ":"))
