import json

import httpx
import streamlit as st

from api_client import (
    create_monitoring_job,
    deactivate_monitoring_job,
    list_alerts,
    list_monitoring_jobs,
    list_projects,
)


st.set_page_config(page_title="Monitoring - Nexus", page_icon="N", layout="wide")
st.title("Monitoring")

try:
    projects = list_projects()
    if not projects:
        st.info("Create a project before adding monitoring jobs.")
        st.stop()

    project = st.selectbox(
        "Project",
        projects,
        format_func=lambda item: item["name"],
    )

    with st.form("monitoring_job"):
        topic = st.text_input("Topic")
        schedule_cron = st.text_input("Cron schedule", value="0 */6 * * *")
        queries_text = st.text_area(
            "Search queries",
            value="latest updates\nregulatory changes\nmarket news",
            height=120,
        )
        submitted = st.form_submit_button("Create monitoring job", use_container_width=True)

    if submitted:
        queries = [line.strip() for line in queries_text.splitlines() if line.strip()]
        payload = {
            "project_id": project["id"],
            "topic": topic.strip(),
            "search_queries": queries,
            "schedule_cron": schedule_cron.strip(),
        }
        create_monitoring_job(payload)
        st.success("Monitoring job created.")
        st.rerun()

    st.subheader("Active jobs")
    jobs = list_monitoring_jobs(project["id"])
    for job in jobs:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.write(f"**{job['topic']}**")
            c1.caption(
                f"Cron: {job['schedule_cron']} | "
                f"Last: {job.get('last_run_at') or 'never'} | "
                f"Next: {job.get('next_run_at') or 'not scheduled'}"
            )
            c1.code(json.dumps(job.get("search_queries", []), indent=2))
            if job.get("is_active") and c2.button("Deactivate", key=job["id"]):
                deactivate_monitoring_job(job["id"])
                st.rerun()

    st.subheader("Alert feed")
    alerts = list_alerts(project["id"])
    if not alerts:
        st.info("No alerts in the last 7 days.")
    for alert in alerts:
        with st.container(border=True):
            st.write(f"**Relevance {float(alert['relevance_score']):.2f}**")
            st.write(alert["summary"])
            st.caption(alert["created_at"])
except httpx.HTTPError as exc:
    st.error(f"Monitoring unavailable: {exc}")
