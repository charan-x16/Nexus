import httpx
import streamlit as st

from api_client import list_projects, search_project_memory


st.set_page_config(page_title="Memory - Nexus", page_icon="N", layout="wide")
st.title("Memory")

try:
    projects = list_projects()
    if not projects:
        st.info("No projects available.")
        st.stop()

    project = st.selectbox(
        "Project",
        projects,
        format_func=lambda item: item["name"],
    )
    query = st.text_input("Search project memory")
    if query.strip():
        results = search_project_memory(project["id"], query.strip())
        if not results:
            st.info("No matching memory chunks found.")
        for result in results:
            with st.container(border=True):
                st.caption(result.get("source_url") or "stored memory")
                st.write(result.get("content", "")[:1200])
                st.progress(min(1.0, max(0.0, float(result.get("score", 0)) / 10)))
except httpx.HTTPError as exc:
    st.error(f"Memory search unavailable: {exc}")
