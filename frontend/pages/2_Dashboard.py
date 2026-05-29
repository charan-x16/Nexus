from typing import Any

import httpx
import streamlit as st

from api_client import (
    get_dashboard_summary,
    get_run_tokens,
    get_run_trace,
)


def money(value: Any) -> str:
    return f"${float(value or 0):.4f}"


st.set_page_config(page_title="Observability - Nexus", page_icon="N", layout="wide")
st.title("Observability")

try:
    summary = get_dashboard_summary()
    cost_rows = summary.get("cost_by_project_last_30_days", [])
    recent_runs = summary.get("recent_runs", [])

    a, b, c = st.columns(3)
    a.metric("Runs this month", summary.get("total_runs_this_month", 0))
    b.metric("Cost this month", money(summary.get("total_cost_this_month")))
    c.metric("Average cost/run", money(summary.get("avg_cost_per_run_this_month")))

    if cost_rows:
        st.subheader("Cost by project, last 30 days")
        st.bar_chart(
            {row["project_name"]: float(row["total_cost"]) for row in cost_rows},
            y_label="USD",
        )

    st.subheader("Recent workflow runs")
    if recent_runs:
        st.dataframe(
            [
                {
                    "project": run["project_name"],
                    "status": run["status"],
                    "run_id": run["id"],
                    "updated_at": run["updated_at"],
                }
                for run in recent_runs
            ],
            use_container_width=True,
        )
        selected_run = st.selectbox(
            "Token breakdown",
            [run["id"] for run in recent_runs],
        )
        tokens = get_run_tokens(selected_run)
        trace = get_run_trace(selected_run)
        st.link_button("Open LangSmith trace search", trace["trace_url"])
        by_agent = tokens.get("by_agent", [])
        if by_agent:
            st.bar_chart(
                {
                    row["agent_name"]: {
                        "input": row["input_tokens"],
                        "output": row["output_tokens"],
                    }
                    for row in by_agent
                }
            )
            st.write(f"Run cost: **{money(tokens.get('total_cost'))}**")
        else:
            st.info("No token usage has been recorded for this run yet.")
    else:
        st.info("No workflow runs yet.")
except httpx.HTTPError as exc:
    st.error(f"Dashboard unavailable: {exc}")
