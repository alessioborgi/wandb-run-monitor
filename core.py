"""Collect completed W&B runs for one creator and aggregate progress."""
import os
import json
from collections import Counter

PROJECT = "adavit/sheaf-mpnn-runs"
TARGET = 250
CREATOR = "fs395"
KEYS = ("state", "model", "dataset", "transport_type")
BASE_URL = "https://api.forge.coreweave.com"


def _user_name(run):
    """Read the creator field from W&B's run metadata."""
    user = (getattr(run, "_attrs", {}) or {}).get("user")
    if not user:
        user = getattr(run, "user", None)
    if isinstance(user, dict):
        return user.get("username")
    return getattr(user, "username", None)


def _config_str(config, key):
    value = (config or {}).get(key, "(missing)")
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def fetch(progress=None):
    """Fetch full run configurations by *page*, not one HTTP request per run.

    User and state filtering happen server-side. The client also validates
    each returned run so counts cannot include the wrong creator or state.
    """
    import wandb

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("WANDB_API_KEY is missing from Streamlit secrets")

    api = wandb.Api(
        api_key=key,
        overrides={"base_url": os.getenv("WANDB_BASE_URL", BASE_URL)},
        timeout=45,
    )
    filters = {"state": "finished", "username": CREATOR}
    runs = api.runs(
        PROJECT,
        filters=filters,
        per_page=100,
        lazy=False,  # important: avoid N additional config fetches
    )
    rows = []
    examined = 0
    unknown = 0
    for run in runs:
        examined += 1
        username = _user_name(run)
        if not username:
            unknown += 1
            continue
        if username.casefold() != CREATOR or str(run.state).lower() != "finished":
            continue

        config = run.config or {}
        rows.append({
            "state": "finished",
            "model": _config_str(config, "model"),
            "dataset": _config_str(config, "dataset"),
            "transport_type": _config_str(config, "transport_type"),
            "run_id": str(run.id),
            "name": str(run.name or run.id),
            "url": str(getattr(run, "url", "") or ""),
        })
        if progress is not None and examined % 50 == 0:
            progress(examined, len(rows))
    if progress is not None:
        progress(examined, len(rows))
    return {"runs": rows, "examined": examined, "unknown": unknown}


def aggregate(runs):
    counts = Counter(tuple(run[k] for k in KEYS) for run in runs)
    return sorted(
        [
            {
                **dict(zip(KEYS, group)),
                "finished": count,
                "remaining": max(0, TARGET - count),
                "progress": min(100.0, 100.0 * count / TARGET),
                "status": "Complete" if count >= TARGET else
                          "Almost there" if count >= 200 else "In progress",
            }
            for group, count in counts.items()
        ],
        key=lambda row: (-row["finished"], row["model"], row["dataset"], row["transport_type"]),
    )


def group_id(row):
    return json.dumps([row[k] for k in KEYS], separators=(",", ":"))
