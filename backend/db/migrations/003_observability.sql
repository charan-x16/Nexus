CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS token_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES workflow_runs(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    search_queries JSONB NOT NULL DEFAULT '[]'::JSONB,
    schedule_cron TEXT NOT NULL,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES monitoring_jobs(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    new_findings TEXT NOT NULL,
    relevance_score NUMERIC(3, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_token_usage_run_id ON token_usage(run_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_agent_name ON token_usage(agent_name);
CREATE INDEX IF NOT EXISTS idx_monitoring_jobs_project_id ON monitoring_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_jobs_active ON monitoring_jobs(is_active);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_job_id ON monitoring_alerts(job_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_created_at ON monitoring_alerts(created_at);
