-- ══════════════════════════════════════════════════════════
--  NEXORA — Supabase Database Schema
--  Run this in Supabase SQL Editor (Dashboard → SQL → New Query)
-- ══════════════════════════════════════════════════════════

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ────────────────────────────────────────────────────────
--  1. USERS TABLE
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           TEXT UNIQUE NOT NULL,
    name            TEXT DEFAULT '',
    role            TEXT DEFAULT 'student' CHECK (role IN ('student', 'tpc')),
    skill_profile   JSONB DEFAULT '{}'::jsonb,
    coach_memory    JSONB DEFAULT '[]'::jsonb,
    readiness_score INTEGER DEFAULT 0 CHECK (readiness_score >= 0 AND readiness_score <= 100),
    interview_scores JSONB DEFAULT '[]'::jsonb,
    at_risk         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_at_risk ON users(at_risk) WHERE at_risk = TRUE;

-- ────────────────────────────────────────────────────────
--  2. TASKS TABLE
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tasks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    due_date    TIMESTAMPTZ,
    status      TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'overdue')),
    assigned_by TEXT DEFAULT 'agent' CHECK (assigned_by IN ('agent', 'manual', 'coach', 'system')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);

-- ────────────────────────────────────────────────────────
--  3. INTERVIEW SESSIONS TABLE
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interview_sessions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    questions   JSONB DEFAULT '[]'::jsonb,
    answers     JSONB DEFAULT '[]'::jsonb,
    scores      JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interview_sessions_user_id ON interview_sessions(user_id);

-- ────────────────────────────────────────────────────────
--  4. ALERTS TABLE
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    severity     TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    message      TEXT NOT NULL,
    sent_at      TIMESTAMPTZ DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_unacknowledged ON alerts(acknowledged) WHERE acknowledged = FALSE;

-- ────────────────────────────────────────────────────────
--  5. ROADMAP TABLE
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS roadmap (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    weeks        JSONB NOT NULL DEFAULT '[]'::jsonb,
    generated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roadmap_user_id ON roadmap(user_id);

-- ────────────────────────────────────────────────────────
--  6. ROW LEVEL SECURITY (RLS)
-- ────────────────────────────────────────────────────────

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE interview_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE roadmap ENABLE ROW LEVEL SECURITY;

-- Users: Students can read their own data, TPC can read all
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid() = id);

-- Service role bypass (for backend operations)
CREATE POLICY "Service role full access users" ON users
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access tasks" ON tasks
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access interview_sessions" ON interview_sessions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access alerts" ON alerts
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access roadmap" ON roadmap
    FOR ALL USING (auth.role() = 'service_role');

-- Students: read own tasks
CREATE POLICY "Students can view own tasks" ON tasks
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Students can view own interviews" ON interview_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Students can view own alerts" ON alerts
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Students can view own roadmap" ON roadmap
    FOR SELECT USING (auth.uid() = user_id);

-- ══════════════════════════════════════════════════════════
--  DONE — Schema ready for NEXORA
-- ══════════════════════════════════════════════════════════
