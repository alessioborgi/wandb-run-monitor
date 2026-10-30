"""Interactive, protected W&B run progress dashboard."""
import os
import re
import hmac
import streamlit as st
import pandas as pd
import plotly.express as px

from core import PROJECT, CREATOR, TARGET, fetch, aggregate

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
def read(_progress=None):
    return fetch(progress=_progress)


if refresh:
    read.clear()


@st.fragment(run_every="120s" if auto else None)
def dashboard():
    with st.status("Loading finished runs from W&B…", expanded=True) as status:
        step = st.empty()

        def on_progress(examined, matched):
            step.write(f"Fetched {examined:,} runs; {matched:,} passed the creator verification…")

        try:
            data = read(_progress=on_progress)
        except Exception as exc:
            status.update(label="Could not load W&B data", state="error", expanded=True)
            # Never accidentally echo a credential from a lower-level exception.
            error = str(exc)
            for secret_name in ("WANDB_API_KEY", "DASHBOARD_PASSWORD"):
                secret = os.environ.get(secret_name, "")
                if secret:
                    error = error.replace(secret, "[REDACTED]")
            step.error(f"{type(exc).__name__}: {error}")
            st.info(
                "Check WANDB_API_KEY and WANDB_BASE_URL in Streamlit app secrets. "
                "The API endpoint should be https://api.forge.coreweave.com, "
                "not a W&B dashboard URL. A large project may take time on first load."
            )
            return
        step.write(
            f'Finished fetching: {data["examined"]:,} server-filtered runs; '
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

    st.dataframe(
        filtered,
        hide_index=True,
        use_container_width=True,
        column_config={
            "progress": st.column_config.ProgressColumn(
                "Completion", min_value=0, max_value=100, format="%.1f%%"
            ),
            "finished": st.column_config.NumberColumn("Finished"),
            "remaining": st.column_config.NumberColumn("Remaining"),
        },
    )
    st.download_button(
        "Download group counts (CSV)",
        filtered.to_csv(index=False),
        file_name="wandb_progress.csv",
        mime="text/csv",
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
    st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
    st.download_button(
        "Download selected runs (CSV)",
        pd.DataFrame(details).to_csv(index=False),
        file_name="wandb_runs.csv",
        mime="text/csv",
    )


dashboard()
