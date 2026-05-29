import time
from typing import Any

import httpx
import streamlit as st

from api_client import (
    approve_workflow,
    cancel_workflow,
    create_workflow,
    fetch_workflow,
    list_projects,
    reject_workflow,
)


def render_plan(plan: dict[str, Any]) -> None:
    st.subheader(plan.get("title", "Research plan"))
    st.caption(plan.get("goal", ""))
    for task in plan.get("subtasks", []):
        label = f"{task.get('priority', '-')}. {task.get('description', 'Research task')}"
        with st.expander(label, expanded=True):
            for query in task.get("search_queries", []):
                st.write(f"- {query}")


def render_cost(cost: dict[str, Any] | None) -> None:
    if not cost:
        return
    st.info(
        "Estimated workflow cost: "
        f"${float(cost.get('estimated_usd', 0)):.4f} "
        f"(range ${float(cost.get('min_usd', 0)):.4f}-"
        f"${float(cost.get('max_usd', 0)):.4f})"
    )


def render_research(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("Research is running.")
        return
    st.subheader("Research")
    for result in results:
        title = result.get("title") or result.get("url", "")
        score = float(result.get("relevance_score", 0))
        with st.expander(f"{score:.1f}/10 - {title}"):
            st.caption(result.get("url", ""))
            st.write(result.get("content", "")[:1400])


def render_critic(reports: list[dict[str, Any]]) -> None:
    if not reports:
        return
    latest = reports[-1]
    findings = latest.get("findings", [])
    st.write(
        f"Critic loop: **Iteration {latest.get('iteration', len(reports))}/3** - "
        f"**{len(findings)}** issues found"
    )
    with st.expander("Critic findings", expanded=bool(findings)):
        if not findings:
            st.success(latest.get("recommendation", "Research passed."))
        for finding in findings:
            severity = finding.get("severity", "low")
            message = (
                f"**{severity.upper()}** {finding.get('finding_type', 'finding')}: "
                f"{finding.get('description', '')}"
            )
            if severity == "high":
                st.error(message)
            elif severity == "medium":
                st.warning(message)
            else:
                st.info(message)


def render_report(report: dict[str, Any]) -> None:
    score = float(report.get("confidence_score", 0))
    st.metric("Confidence", f"{score:.2f}")
    st.title(report.get("title", "Final Report"))
    st.info(report.get("executive_summary", ""))
    for section in report.get("sections", []):
        st.header(section.get("title", "Section"))
        st.markdown(section.get("content", ""))
    citations = report.get("all_citations", [])
    if citations:
        st.subheader("References")
        for citation in sorted(citations, key=lambda item: item.get("index", 0)):
            st.markdown(
                f"[{citation.get('index')}] **{citation.get('title') or citation.get('url')}**  \n"
                f"{citation.get('url')}  \n"
                f"> {citation.get('quote', '')}"
            )


st.set_page_config(page_title="Workflows - Nexus", page_icon="N", layout="wide")
st.title("Workflows")

for key, default in {
    "phase5_run_id": None,
    "phase5_plan": None,
    "phase5_cost": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

try:
    projects = list_projects()
except httpx.HTTPError as exc:
    projects = []
    st.error(f"Could not load projects: {exc}")

project_labels = ["New project"] + [project["name"] for project in projects]
selected_label = st.selectbox("Project", project_labels)
selected_project = None
if selected_label != "New project":
    selected_project = next(project for project in projects if project["name"] == selected_label)

with st.form("create_workflow"):
    project_name = st.text_input(
        "Project name",
        value=selected_project["name"] if selected_project else "Research Brief",
        disabled=selected_project is not None,
    )
    goal = st.text_area("Goal", height=160)
    submitted = st.form_submit_button("Create plan", use_container_width=True)

if submitted:
    try:
        created = create_workflow(
            goal=goal.strip(),
            project_name=project_name.strip(),
            project_id=selected_project["id"] if selected_project else None,
        )
        st.session_state.phase5_run_id = created["run_id"]
        st.session_state.phase5_plan = created.get("plan")
        st.session_state.phase5_cost = created.get("cost_estimate")
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Could not create workflow: {exc}")

run_id = st.session_state.phase5_run_id
if run_id:
    st.divider()
    st.caption(f"Run {run_id}")
    try:
        workflow = fetch_workflow(run_id)
        status = workflow.get("status", "unknown")
        progress = {
            "planning": 0.1,
            "awaiting_approval": 0.25,
            "queued": 0.35,
            "researching": 0.6,
            "criticizing": 0.75,
            "targeted_research": 0.8,
            "writing": 0.9,
            "completed": 1.0,
            "rejected": 1.0,
            "cancelled": 1.0,
            "failed": 1.0,
        }.get(status, 0.2)
        st.progress(progress)
        st.write(f"Status: **{status}**")

        plan = workflow.get("plan") or st.session_state.phase5_plan
        render_cost(st.session_state.phase5_cost)
        if plan:
            render_plan(plan)

        if status == "awaiting_approval":
            approve_col, reject_col = st.columns(2)
            if approve_col.button("Approve Plan", use_container_width=True):
                approve_workflow(run_id)
                st.rerun()
            if reject_col.button("Reject Plan", use_container_width=True):
                reject_workflow(run_id)
                st.rerun()
        elif status in {"researching", "criticizing", "targeted_research", "writing"}:
            if st.button("Cancel run"):
                cancel_workflow(run_id)
                st.rerun()

        render_critic(workflow.get("critic_reports", []))
        if status in {"researching", "criticizing", "targeted_research", "writing", "completed"}:
            render_research(workflow.get("research_results", []))

        if status == "completed" and workflow.get("final_report"):
            st.divider()
            render_report(workflow["final_report"])
        elif status == "failed":
            st.error(workflow.get("final_output") or "Workflow failed.")
        elif status in {"researching", "criticizing", "targeted_research", "writing"}:
            time.sleep(2)
            st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Could not fetch workflow: {exc}")
