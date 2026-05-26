import asyncio
import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


async def create_workflow(goal: str, project_name: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.post(
            "/workflows",
            json={"goal": goal, "project_name": project_name},
        )
        response.raise_for_status()
        return response.json()


async def fetch_workflow(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        response = await client.get(f"/workflows/{run_id}")
        response.raise_for_status()
        return response.json()


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


st.set_page_config(page_title="Nexus", page_icon="N", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 980px;
        padding-top: 2.25rem;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0;
    }
    .nexus-title {
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: 2.6rem;
        line-height: 1.05;
        margin-bottom: 0.3rem;
    }
    .nexus-subtitle {
        color: #5d6472;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nexus-title">Nexus</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="nexus-subtitle">Plan, write, and track one focused knowledge-work run.</div>',
    unsafe_allow_html=True,
)

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "final_output" not in st.session_state:
    st.session_state.final_output = None

with st.form("workflow_form", clear_on_submit=False):
    project_name = st.text_input("Project name", value="Phase 1 Research Brief")
    goal = st.text_area(
        "Goal",
        height=180,
    )
    submitted = st.form_submit_button("Start workflow", use_container_width=True)

if submitted:
    if not goal.strip() or not project_name.strip():
        st.error("Both project name and goal are required.")
    else:
        try:
            created = run_async(create_workflow(goal.strip(), project_name.strip()))
            st.session_state.run_id = created["run_id"]
            st.session_state.final_output = None
            st.rerun()
        except httpx.HTTPError as exc:
            st.error(f"Could not start workflow: {exc}")

if st.session_state.run_id:
    st.divider()
    try:
        workflow = run_async(fetch_workflow(st.session_state.run_id))
        status = workflow["status"]
        st.caption(f"Run {st.session_state.run_id}")
        st.progress(
            {"queued": 0.15, "running": 0.55, "completed": 1.0, "failed": 1.0}.get(
                status,
                0.25,
            )
        )
        st.write(f"Status: **{status}**")

        if status == "completed" and workflow.get("final_output"):
            st.session_state.final_output = workflow["final_output"]
        elif status == "failed":
            state = workflow.get("state", {})
            st.error(state.get("final_output", "Workflow failed."))
        else:
            time.sleep(2)
            st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Could not fetch workflow status: {exc}")

if st.session_state.final_output:
    st.divider()
    st.markdown(st.session_state.final_output)
