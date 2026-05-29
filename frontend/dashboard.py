import httpx
import streamlit as st

from api_client import API_BASE_URL, list_projects


st.set_page_config(page_title="Nexus Dashboard", page_icon="N", layout="wide")

st.title("Nexus")
st.caption("AI Operating System for knowledge work")

try:
    projects = list_projects()
    st.metric("Projects", len(projects))
    st.write(f"Backend: `{API_BASE_URL}`")
except httpx.HTTPError as exc:
    st.error(f"Backend unavailable: {exc}")

st.page_link("pages/1_Workflows.py", label="Workflows")
st.page_link("pages/2_Dashboard.py", label="Observability")
st.page_link("pages/3_Monitoring.py", label="Monitoring")
st.page_link("pages/4_Memory.py", label="Memory")
