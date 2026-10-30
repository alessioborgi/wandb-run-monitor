"""Low-memory Streamlit interface for group-level W&B monitoring."""
import os
import hmac

import pandas as pd
import plotly.express as px
import streamlit as st

from core import CREATOR, PROJECT, TARGET, ROC_DATASETS, TEST_COLUMNS, fetch

st.set_page_config(page_title="Sheaf Run Monitor", page_icon="📈", layout="wide")
st.title("Sheaf Run Monitor")
st.caption(f"Project: {PROJECT}  ·  State: finished  ·  User: {CREATOR}  ·  Target: {TARGET} runs")

password = os.environ.get("DASHBOARD_PASSWORD", "")
if not password:
    st.error("DASHBOARD_PASSWORD is not configured in Streamlit app secrets.")
    st.stop()
if not st.session_state.get("authorized", False):
    with st.form("login"):
        attempt = st.text_input("Dashboard password", type="password")
        submitted = st.form_submit_button("Unlock dashboard")
    if submitted:
        if hmac.compare_digest(attempt, password):
            st.session_state.authorized = True
            st.rerun()
        st.error("Incorrect password")
    st.stop()

with st.sidebar:
    st.header("Scope")
    st.text("Project: " + PROJECT)
    st.text("State: finished")
    st.text("User: " + CREATOR)
    st.caption("Grouped by state, model, dataset and transport_type")
    st.divider()
    auto = st.toggle("Auto refresh (5 min)", value=True)
    refresh = st.button("Refresh now", use_container_width=True)
    if st.button("Log out", use_container_width=True):
        st.session_state.authorized = False
        st.rerun()


@st.cache_data(ttl=290, max_entries=1, show_spinner=False)
def read():
    # This cache holds only dozens of compact group summaries, not W&B Run objects.
    return fetch()


if refresh:
    read.clear()


@st.fragment(run_every="300s" if auto else None)
def dashboard():
    with st.status("Reading compact W&B result pages…", expanded=True) as status:
        try:
            data = read()
        except Exception as exc:
            status.update(label="W&B request failed", state="error", expanded=True)
            error = str(exc)
            for name in ("WANDB_API_KEY", "DASHBOARD_PASSWORD"):
                secret = os.environ.get(name, "")
                if secret:
                    error = error.replace(secret, "[REDACTED]")
            st.error(f"{type(exc).__name__}: {error}")
            st.info("Check your W&B API endpoint and app logs. No run data was deleted.")
            return
        status.update(label="W&B snapshot ready", state="complete", expanded=False)

    groups = data["groups"]
    st.caption(
        f'Last updated: {data["fetched_at"]} · '
        f'Examined {data["examined"]:,} runs; {data["matched"]:,} matched user {CREATOR}. '
        "Only compact group summaries are retained in memory."
    )
    if data["unknown"]:
        st.warning(
            f'{data["unknown"]:,} returned runs lacked creator metadata and were excluded.'
        )

    a, b, c, d = st.columns(4)
    a.metric("Finished runs", f'{data["matched"]:,}')
    b.metric("Combinations", len(groups))
    c.metric("Reached 250", sum(g["finished"] >= TARGET for g in groups))
    d.metric("Runs remaining", sum(g["remaining"] for g in groups))
    if not groups:
        st.warning("No finished runs matched the creator. Check W&B permissions and filters.")
        return

    df = pd.DataFrame(groups)
    st.subheader("Progress per combination")
    cols = st.columns(3)
    filtered = df.copy()
    for col, field in zip(cols, ("model", "dataset", "transport_type")):
        selected = col.multiselect(
            field.replace("_", " ").title(),
            sorted(df[field].unique()),
            key="filter_" + field,
        )
        if selected:
            filtered = filtered.loc[filtered[field].isin(selected)]
    if filtered.empty:
        st.info("No combinations match your filters.")
        return

    is_roc = filtered["dataset"].str.strip().str.casefold().isin(ROC_DATASETS)
    score_cols = []
    if (~is_roc).any():
        score_cols += ["mean_test_acc", "std_test_acc"]
    if is_roc.any():
        score_cols += ["mean_test_roc_auc", "std_test_roc_auc"]

    columns = [
        "state", "model", "dataset", "transport_type",
        "finished", "remaining", "progress", "status",
        "selection_metric", "best_validation",
        "valid_validation_runs", *score_cols, "best_run_url",
    ]

    def percent_frame(frame):
        output = frame.copy()
        for name in score_cols:
            output[name] = pd.to_numeric(output[name], errors="coerce") * 100
        return output

    score_formats = {
        name: st.column_config.NumberColumn(name + " (%)", format="%.2f%%")
        for name in score_cols
    }

    st.dataframe(
        percent_frame(filtered[columns]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "progress": st.column_config.ProgressColumn(
                "Completion", min_value=0, max_value=100, format="%.1f%%"
            ),
            "best_validation": st.column_config.NumberColumn(
                "Best validation", format="%.5f"
            ),
            "best_run_url": st.column_config.LinkColumn(
                "Selected run", display_text="Open run"
            ),
            **score_formats,
        },
    )
    st.caption(
        "The best run is chosen by maximum mean_val_acc, except minesweeper, "
        "tolokers and questions (maximum mean_val_roc_auc). Test mean and "
        "standard deviation are displayed in percent from that same run."
    )

    missing = filtered.loc[filtered["best_validation"].isna()]
    if not missing.empty:
        st.warning(
            f"{len(missing)} combination(s) have no valid validation metric in any "
            "finished run. This is a count of combinations, not individual runs."
        )
        with st.expander("Identify combinations with missing validation scores"):
            diag = missing[[
                "model", "dataset", "transport_type", "finished",
                "selection_metric", "valid_validation_runs",
                "valid_all_metrics_runs", "available_metric_keys",
            ]].copy()
            diag["available_metric_keys"] = diag["available_metric_keys"].apply(
                lambda x: ", ".join(x) if x else "(none found)"
            )
            st.dataframe(diag, hide_index=True, use_container_width=True)

    st.subheader("Best results by validation")
    leaderboard = filtered.sort_values(
        ["dataset", "best_validation", "model", "transport_type"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    st.dataframe(
        percent_frame(leaderboard[[
            "dataset", "model", "transport_type", "selection_metric",
            "best_validation", *score_cols, "finished", "best_run_url",
        ]]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "best_validation": st.column_config.NumberColumn("Best validation", format="%.5f"),
            "best_run_url": st.column_config.LinkColumn("Best run", display_text="Open run"),
            **score_formats,
        },
    )

    st.subheader("Progress overview")
    chart = filtered[["model", "dataset", "transport_type", "finished", "remaining"]].copy()
    chart["Combination"] = (
        chart["model"] + " / " + chart["dataset"] + " / " + chart["transport_type"]
    )
    chart = chart.nlargest(25, "finished")
    fig = px.bar(
        chart, x="finished", y="Combination", orientation="h",
        hover_data=["remaining"], title="Combinations closest to 250 runs"
    )
    fig.add_vline(x=TARGET, line_dash="dash", annotation_text="Target 250")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(360, 30 * len(chart)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inspect a combination")
    selected_id = st.selectbox(
        "Combination",
        filtered.index.tolist(),
        format_func=lambda i: (
            f'{df.loc[i, "model"]} / {df.loc[i, "dataset"]} / '
            f'{df.loc[i, "transport_type"]} ({df.loc[i, "finished"]}/{TARGET})'
        ),
    )
    selected = df.loc[selected_id]
    st.write(
        f'Valid validation runs: {selected["valid_validation_runs"]:,}; '
        f'complete-metric runs: {selected["valid_all_metrics_runs"]:,}; '
        f'runs with missing/invalid fields: {selected["invalid_metric_runs"]:,}.'
    )
    st.caption(
        "Showing up to five example runs per group to control memory use. "
        "Open the selected W&B run or the main W&B project for the complete list."
    )
    samples = selected["invalid_samples"]
    if samples:
        st.write("Examples of runs missing required metrics")
        st.dataframe(pd.DataFrame(samples), hide_index=True, use_container_width=True)
    else:
        st.dataframe(
            pd.DataFrame(selected["sample_runs"]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Export")
    export = filtered.drop(
        columns=["invalid_samples", "sample_runs", "available_metric_keys"]
    ).copy()
    for name in TEST_COLUMNS:
        export[name + "_pct"] = pd.to_numeric(export[name], errors="coerce") * 100
        export = export.drop(columns=name)
    st.download_button(
        "Download all group summaries (CSV)",
        export.to_csv(index=False),
        file_name="wandb_group_monitor.csv",
        mime="text/csv",
    )


dashboard()
