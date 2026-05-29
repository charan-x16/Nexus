import asyncio
import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


async def list_projects() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.get("/projects")
        response.raise_for_status()
        return response.json()


async def list_project_runs(project_id: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.get(f"/projects/{project_id}/runs")
        response.raise_for_status()
        return response.json()


async def search_project_memory(project_id: str, query: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=45.0) as client:
        response = await client.get(
            f"/projects/{project_id}/memory",
            params={"query": query},
        )
        response.raise_for_status()
        return response.json()


async def create_workflow(
    goal: str,
    project_name: str,
    project_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"goal": goal, "project_name": project_name}
    if project_id:
        payload["project_id"] = project_id
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0) as client:
        response = await client.post("/workflows", json=payload)
        response.raise_for_status()
        return response.json()


async def approve_workflow(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.post(f"/workflows/{run_id}/approve")
        response.raise_for_status()
        return response.json()


async def reject_workflow(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.post(f"/workflows/{run_id}/reject")
        response.raise_for_status()
        return response.json()


async def fetch_workflow(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.get(f"/workflows/{run_id}/status")
        response.raise_for_status()
        return response.json()


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def reset_workflow() -> None:
    st.session_state.run_id = None
    st.session_state.plan = None
    st.session_state.status = None
    st.session_state.final_output = None


def render_plan(plan: dict[str, Any]) -> None:
    st.subheader(plan.get("title", "Research plan"))
    st.caption(plan.get("goal", ""))
    for task in plan.get("subtasks", []):
        label = f"{task.get('priority', '-')}. {task.get('description', 'Research task')}"
        with st.expander(label, expanded=True):
            for query in task.get("search_queries", []):
                st.write(f"- {query}")


def render_research_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("Research is running.")
        return

    st.subheader("Research")
    for result in results:
        url = result.get("url", "")
        title = result.get("title") or url
        score = result.get("relevance_score", 0)
        with st.expander(f"{score}/10 - {title}", expanded=False):
            st.caption(url)
            st.write(result.get("content", "")[:1200])


def render_memory_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        with st.container(border=True):
            st.caption(result.get("source_url") or "stored memory")
            st.write(result.get("content", "")[:900])
            st.progress(min(1.0, max(0.0, float(result.get("score", 0)) / 10)))


st.set_page_config(page_title="Nexus", page_icon="N", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1080px;
        padding-top: 2rem;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0;
    }
    .nexus-title {
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: 2.5rem;
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }
    .nexus-subtitle {
        color: #596272;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "run_id": None,
    "plan": None,
    "status": None,
    "final_output": None,
    "selected_project_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

projects: list[dict[str, Any]] = []
selected_project: dict[str, Any] | None = None

with st.sidebar:
    st.header("Projects")
    try:
        projects = run_async(list_projects())
        project_labels = ["New project"] + [project["name"] for project in projects]
        selected_label = st.selectbox("Project", project_labels)
        if selected_label != "New project":
            selected_project = next(
                project for project in projects if project["name"] == selected_label
            )
            st.session_state.selected_project_id = selected_project["id"]
        else:
            st.session_state.selected_project_id = None

        if selected_project:
            st.subheader("Runs")
            runs = run_async(list_project_runs(selected_project["id"]))
            for run in runs[:6]:
                st.caption(f"{run['status']} - {run['id']}")

            st.subheader("Memory")
            memory_query = st.text_input("Search Memory")
            if memory_query.strip():
                memory_results = run_async(
                    search_project_memory(selected_project["id"], memory_query.strip())
                )
                render_memory_results(memory_results)
    except httpx.HTTPError as exc:
        st.error(f"Project sidebar unavailable: {exc}")

st.markdown('<div class="nexus-title">Nexus</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="nexus-subtitle">Approve the plan, gather research, and preserve memory.</div>',
    unsafe_allow_html=True,
)

with st.form("workflow_form", clear_on_submit=False):
    project_name = st.text_input(
        "Project name",
        value=selected_project["name"] if selected_project else "Research Brief",
        disabled=selected_project is not None,
    )
    goal = st.text_area("Goal", height=170)
    submitted = st.form_submit_button("Create plan", use_container_width=True)

if submitted:
    if not goal.strip() or not project_name.strip():
        st.error("Both project name and goal are required.")
    else:
        try:
            reset_workflow()
            created = run_async(
                create_workflow(
                    goal=goal.strip(),
                    project_name=project_name.strip(),
                    project_id=st.session_state.selected_project_id,
                )
            )
            st.session_state.run_id = created["run_id"]
            st.session_state.status = created["status"]
            st.session_state.plan = created.get("plan")
            st.rerun()
        except httpx.HTTPError as exc:
            st.error(f"Could not create workflow: {exc}")

if st.session_state.run_id:
    st.divider()
    st.caption(f"Run {st.session_state.run_id}")

    try:
        workflow = run_async(fetch_workflow(st.session_state.run_id))
        st.session_state.status = workflow["status"]
        st.session_state.plan = workflow.get("plan") or st.session_state.plan
        st.session_state.final_output = workflow.get("final_output")

        progress = {
            "planning": 0.1,
            "awaiting_approval": 0.25,
            "researching": 0.6,
            "completed": 1.0,
            "rejected": 1.0,
            "failed": 1.0,
        }.get(st.session_state.status, 0.2)
        st.progress(progress)
        st.write(f"Status: **{st.session_state.status}**")

        if st.session_state.plan:
            render_plan(st.session_state.plan)

        if st.session_state.status == "awaiting_approval":
            approve_col, reject_col = st.columns(2)
            with approve_col:
                if st.button("Approve Plan", use_container_width=True):
                    run_async(approve_workflow(st.session_state.run_id))
                    st.session_state.status = "researching"
                    st.rerun()
            with reject_col:
                if st.button("Reject Plan", use_container_width=True):
                    run_async(reject_workflow(st.session_state.run_id))
                    st.session_state.status = "rejected"
                    st.rerun()

        research_results = workflow.get("research_results", [])
        if st.session_state.status in {"researching", "completed"}:
            render_research_results(research_results)

        if st.session_state.status == "completed" and st.session_state.final_output:
            st.divider()
            st.markdown(st.session_state.final_output)
        elif st.session_state.status == "failed":
            state = workflow.get("state", {})
            st.error(state.get("final_output", "Workflow failed."))
        elif st.session_state.status == "rejected":
            st.warning("Workflow rejected.")
        elif st.session_state.status == "researching":
            time.sleep(2)
            st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Could not fetch workflow status: {exc}")
