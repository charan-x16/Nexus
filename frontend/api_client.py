import os
from typing import Any

import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 45.0,
) -> Any:
    with httpx.Client(base_url=API_BASE_URL, timeout=timeout) as client:
        response = client.request(method, path, json=payload, params=params)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response.json()


def list_projects() -> list[dict[str, Any]]:
    return request_json("GET", "/projects")


def list_project_runs(project_id: str) -> list[dict[str, Any]]:
    return request_json("GET", f"/projects/{project_id}/runs")


def search_project_memory(project_id: str, query: str) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        f"/projects/{project_id}/memory",
        params={"query": query},
        timeout=60.0,
    )


def create_workflow(
    goal: str,
    project_name: str,
    project_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"goal": goal, "project_name": project_name}
    if project_id:
        payload["project_id"] = project_id
    return request_json("POST", "/workflows", payload=payload, timeout=180.0)


def approve_workflow(run_id: str) -> dict[str, Any]:
    return request_json("POST", f"/workflows/{run_id}/approve")


def reject_workflow(run_id: str) -> dict[str, Any]:
    return request_json("POST", f"/workflows/{run_id}/reject")


def cancel_workflow(run_id: str) -> dict[str, Any]:
    return request_json("POST", f"/workflows/{run_id}/cancel")


def fetch_workflow(run_id: str) -> dict[str, Any]:
    return request_json("GET", f"/workflows/{run_id}/status")


def create_monitoring_job(payload: dict[str, Any]) -> dict[str, Any]:
    return request_json("POST", "/monitoring/jobs", payload=payload)


def list_monitoring_jobs(project_id: str | None = None) -> list[dict[str, Any]]:
    params = {"project_id": project_id} if project_id else None
    return request_json("GET", "/monitoring/jobs", params=params)


def deactivate_monitoring_job(job_id: str) -> None:
    request_json("DELETE", f"/monitoring/jobs/{job_id}")


def list_alerts(project_id: str | None = None) -> list[dict[str, Any]]:
    params = {"project_id": project_id} if project_id else None
    return request_json("GET", "/monitoring/alerts", params=params)


def get_project_cost(project_id: str) -> dict[str, Any]:
    return request_json("GET", f"/observability/projects/{project_id}/cost")


def get_dashboard_summary() -> dict[str, Any]:
    return request_json("GET", "/observability/dashboard")


def get_run_tokens(run_id: str) -> dict[str, Any]:
    return request_json("GET", f"/observability/runs/{run_id}/tokens")


def get_run_trace(run_id: str) -> dict[str, Any]:
    return request_json("GET", f"/observability/runs/{run_id}/trace")
