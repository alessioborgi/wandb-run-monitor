"""Interactive, protected W&B run progress dashboard."""
import os
import hmac
import streamlit as st
import pandas as pd
import plotly.express as px

from core import PROJECT, CREATOR, TARGET, fetch, aggregate, invalid_metric_fields

st.set_page_config(page_title="Sheaf Run Monitor", page_icon="📈", layout="wide")
st.title("Sheaf Run Monitor")
st.caption("Project: adavit/sheaf-mpnn-runs  ·  State: finished  ·  User: fs395  ·  Target: 250 runs")

password = os.environ.get("DASHBOARD_PASSWORD", "")
if not password:
    st.error("DASHBOARD_PASSWORD is not configured. Add it to your app's Streamlit Cloud secrets.")
    st.stop()

if not st.session_state.get("authorized", False):
    with st.form("login"):
        entered = st.text_input("Dashboard password", type="password")
        submitted = st.form_submit_button("Unlock dashboard")
    if submitted:
        if hmac.compare_digest(entered, password):
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

with st.sidebar:
    st.header("Filters")
    st.text("Project: adavit/sheaf-mpnn-runs")
    st.text("State: finished")
    st.text("User: fs395")
    st.caption("Grouped by state, model, dataset and transport_type")
    st.divider()
    auto = st.toggle("Auto refresh every 2 min", value=True)
    refresh = st.button("Refresh now", use_container_width=True)
    if st.button("Log out", use_container_width=True):
        st.session_state.authorized = False
        st.rerun()


@st.cache_data(ttl=110, show_spinner=False)
def read():
    # Data cache contains no Streamlit UI calls or callbacks.
    return fetch()


if refresh:
    read.clear()


@st.fragment(run_every="120s" if auto else None)
def dashboard():
    with st.status("Loading finished runs from W&B…", expanded=True) as status:
        try:
            data = read()
        except Exception as exc:
            status.update(label="Could not load W&B data", state="error", expanded=True)
            # Never accidentally echo a credential from a lower-level exception.
            error = str(exc)
            for secret_name in ("WANDB_API_KEY", "DASHBOARD_PASSWORD"):
                secret = os.environ.get(secret_name, "")
                if secret:
                    error = error.replace(secret, "[REDACTED]")
            st.error(f"{type(exc).__name__}: {error}")
            st.info(
                "Check WANDB_API_KEY and WANDB_BASE_URL in Streamlit app secrets. "
                "The API endpoint should be https://api.forge.coreweave.com, "
                "not a W&B dashboard URL. A large project may take time on first load."
            )
            return
        st.write(
            f'Fetched {data["examined"]:,} server-filtered runs; '
            f'{len(data["runs"]):,} verified matches.'
        )
        status.update(label="W&B data loaded", state="complete", expanded=False)

    rows = data["runs"]
    groups = aggregate(rows)
    if data["unknown"]:
        st.warning(
            f'{data["unknown"]:,} returned runs lacked identifiable creator metadata and were excluded.'
        )

    a, b, c, d = st.columns(4)
    a.metric("Finished runs", f"{len(rows):,}")
    b.metric("Combinations", len(groups))
    c.metric("Reached 250", sum(g["finished"] >= TARGET for g in groups))
    d.metric("Runs remaining", sum(g["remaining"] for g in groups))

    if not groups:
        st.warning(
            "No matching finished runs were returned for creator fs395. "
            "Check the W&B project, creator filter, and access permissions."
        )
        return

    df = pd.DataFrame(groups)
    st.subheader("Progress per combination")
    filter_cols = st.columns(3)
    filtered = df.copy()
    for column, field in zip(filter_cols, ("model", "dataset", "transport_type")):
        selected = column.multiselect(
            field.replace("_", " ").title(),
            sorted(df[field].unique()),
            key="filter_" + field,
        )
        if selected:
            filtered = filtered[filtered[field].isin(selected)]

    if filtered.empty:
        st.info("No combinations match those selections.")
        return

    # Keep the requested test metric names; hide irrelevant pairs when the
    # current selection contains only accuracy or only ROC-AUC datasets.
    roc_datasets = {"minesweeper", "tolokers", "questions"}
    roc_rows = filtered["dataset"].str.strip().str.casefold().isin(roc_datasets)
    score_columns = []
    if (~roc_rows).any():
        score_columns.extend(["mean_test_acc", "std_test_acc"])
    if roc_rows.any():
        score_columns.extend(["mean_test_roc_auc", "std_test_roc_auc"])

    base_columns = [
        "state", "model", "dataset", "transport_type",
        "finished", "remaining", "progress", "status",
        "selection_metric", "best_validation",
        "valid_validation_runs", "valid_all_metrics_runs",
    ]
    display_columns = base_columns + score_columns + ["best_run_url"]

    # Presentation only: W&B stores 0..1 ratios. The displayed test means
    # and standard deviations are both multiplied by 100 (percentage points).
    display_df = filtered[display_columns].copy()
    for metric_name in score_columns:
        display_df[metric_name] = pd.to_numeric(display_df[metric_name], errors="coerce") * 100

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "progress": st.column_config.ProgressColumn(
                "Completion", min_value=0, max_value=100, format="%.1f%%"
            ),
            "finished": st.column_config.NumberColumn("Finished"),
            "remaining": st.column_config.NumberColumn("Remaining"),
            "best_validation": st.column_config.NumberColumn("Best validation", format="%.5f"),
            "mean_test_acc": st.column_config.NumberColumn("mean_test_acc (%)", format="%.2f%%"),
            "std_test_acc": st.column_config.NumberColumn("std_test_acc (%)", format="%.2f%%"),
            "mean_test_roc_auc": st.column_config.NumberColumn("mean_test_roc_auc (%)", format="%.2f%%"),
            "std_test_roc_auc": st.column_config.NumberColumn("std_test_roc_auc (%)", format="%.2f%%"),
            "valid_validation_runs": st.column_config.NumberColumn("Valid validation runs"),
            "valid_all_metrics_runs": st.column_config.NumberColumn("Complete-metric runs"),
            "best_run_url": st.column_config.LinkColumn("Selected run", display_text="Open run"),
        },
    )
    st.caption(
        "Best results are selected by the highest mean_val_acc, except for "
        "minesweeper, tolokers and questions (highest mean_val_roc_auc). "
        "Each mean/std test pair comes from the same selected run. "
        "Missing validation metrics are shown as blank rather than selecting by test."
    )
    missing_best = filtered["best_validation"].isna().sum()
    if missing_best:
        st.warning(
            f"{missing_best} combination(s) have no valid validation metric in any finished run; "
            "their best-result fields are blank. This does NOT imply only "
            f"{missing_best} individual runs are affected."
        )
        with st.expander("Inspect missing validation metrics and available W&B keys"):
            missing_rows = filtered[filtered["best_validation"].isna()].copy()
            missing_rows["available_metric_keys"] = missing_rows["available_metric_keys"].apply(
                lambda keys: ", ".join(keys) if keys else "(none detected)"
            )
            st.dataframe(
                missing_rows[[
                    "model", "dataset", "transport_type", "finished",
                    "selection_metric", "valid_validation_runs",
                    "valid_all_metrics_runs", "available_metric_keys",
                ]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Metrics are read from W&B run summaries. Missing values may "
                "be stored under different names or in run history. "
                "Review runs before deleting anything in W&B."
            )
    invalid_count = int(filtered["invalid_metric_runs"].sum())
    if invalid_count:
        st.info(
            f"{invalid_count:,} finished runs in the selected combinations lack at least "
            "one required validation, test mean, or test std metric. "
            "These runs are still present in W&B and counted as finished."
        )
        with st.expander("Review individual runs with missing or invalid metrics"):
            group_keys = set(
                tuple(row[key] for key in ("state", "model", "dataset", "transport_type"))
                for _, row in filtered.iterrows()
            )
            invalid_rows = [
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "dataset": run["dataset"],
                    "transport_type": run["transport_type"],
                    "missing_or_invalid": ", ".join(invalid_metric_fields(run)),
                    "url": run["url"],
                }
                for run in rows
                if tuple(run[key] for key in ("state", "model", "dataset", "transport_type")) in group_keys
                and invalid_metric_fields(run)
            ]
            st.dataframe(pd.DataFrame(invalid_rows), hide_index=True, use_container_width=True)
            st.download_button(
                "Download run-level invalid metric audit (CSV)",
                pd.DataFrame(invalid_rows).to_csv(index=False),
                file_name="invalid_wandb_runs_audit.csv",
                mime="text/csv",
                key="audit_download",
            )
    # Export unambiguous percentage-point columns rather than silently
    # changing the numeric scale of columns named like W&B raw metrics.
    export_df = filtered.drop(columns=["available_metric_keys"]).copy()
    percentage_names = {}
    for metric_name in ("mean_test_acc", "std_test_acc", "mean_test_roc_auc", "std_test_roc_auc"):
        export_df[metric_name] = pd.to_numeric(export_df[metric_name], errors="coerce") * 100
        percentage_names[metric_name] = metric_name + "_pct"
    export_df = export_df.rename(columns=percentage_names)
    st.download_button(
        "Download group counts and best scores (CSV, test values in %)",
        export_df.to_csv(index=False),
        file_name="wandb_progress_and_best_results_percent.csv",
        mime="text/csv",
    )

    st.subheader("Best results by validation")
    st.caption(
        "Ranked within each dataset by its chosen validation metric, descending. "
        "All model families from fs395 are eligible."
    )
    leaderboard = filtered.sort_values(
        ["dataset", "best_validation", "model", "transport_type"],
        ascending=[True, False, True, True],
        na_position="last",
        kind="stable",
    )
    leaderboard_columns = [
        "dataset", "model", "transport_type", "selection_metric",
        "best_validation", *score_columns, "finished", "best_run_url",
    ]
    leaderboard_display = leaderboard[leaderboard_columns].copy()
    for metric_name in score_columns:
        leaderboard_display[metric_name] = (
            pd.to_numeric(leaderboard_display[metric_name], errors="coerce") * 100
        )
    score_format = {
        name: st.column_config.NumberColumn(name + " (%)", format="%.2f%%")
        for name in score_columns
    }
    st.dataframe(
        leaderboard_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "best_validation": st.column_config.NumberColumn("Best validation", format="%.5f"),
            "best_run_url": st.column_config.LinkColumn("Best run", display_text="Open run"),
            **score_format,
        },
    )

    chart = filtered.copy()
    chart["Combination"] = (
        chart["model"] + " / " + chart["dataset"] + " / " + chart["transport_type"]
    )
    chart = chart.sort_values("finished", ascending=False).head(30)
    fig = px.bar(
        chart,
        x="finished",
        y="Combination",
        orientation="h",
        hover_data=["remaining", "progress", "status"],
        title="Combinations closest to 250 finished runs",
    )
    fig.add_vline(x=TARGET, line_dash="dash", annotation_text="Target: 250")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(360, 30 * len(chart)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inspect individual runs")
    selected_index = st.selectbox(
        "Select a combination",
        filtered.index.tolist(),
        format_func=lambda i: (
            f'{df.loc[i, "model"]} / {df.loc[i, "dataset"]} / '
            f'{df.loc[i, "transport_type"]} ({df.loc[i, "finished"]}/{TARGET})'
        ),
    )
    selected_group = df.loc[selected_index]
    details = [
        run for run in rows
        if all(
            run[k] == selected_group[k]
            for k in ("model", "dataset", "transport_type")
        )
    ]
    details_df = pd.DataFrame(details)
    details_df["missing_or_invalid"] = [
        ", ".join(invalid_metric_fields(run)) for run in details
    ]
    for metric_name in ("mean_test_acc", "std_test_acc", "mean_test_roc_auc", "std_test_roc_auc"):
        details_df[metric_name + "_pct"] = (
            pd.to_numeric(details_df[metric_name], errors="coerce") * 100
        )
    visible_details = [
        "run_id", "name", "model", "dataset", "transport_type",
        "mean_val_acc", "mean_val_roc_auc",
        "mean_test_acc_pct", "std_test_acc_pct",
        "mean_test_roc_auc_pct", "std_test_roc_auc_pct",
        "missing_or_invalid", "url",
    ]
    st.dataframe(details_df[visible_details], hide_index=True, use_container_width=True)
    st.download_button(
        "Download selected runs (CSV)",
        pd.DataFrame(details).to_csv(index=False),
        file_name="wandb_runs.csv",
        mime="text/csv",
    )


dashboard()
