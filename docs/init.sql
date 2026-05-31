-- Database initialization for Video Analytics MVP
-- Run this on Railway PostgreSQL if needed (SQLAlchemy auto-creates tables)

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    venue_id VARCHAR(50) NOT NULL,
    pseudo_id VARCHAR(32) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    zone VARCHAR(50),
    dwell_seconds FLOAT DEFAULT 0,
    age_bracket VARCHAR(10),
    gender VARCHAR(1),
    is_repeat BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS venues (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    api_key VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    config JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    venue_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_visitors INTEGER DEFAULT 0,
    unique_visitors INTEGER DEFAULT 0,
    repeat_visitors INTEGER DEFAULT 0,
    avg_dwell_seconds FLOAT DEFAULT 0,
    peak_hour INTEGER,
    gender_male INTEGER DEFAULT 0,
    gender_female INTEGER DEFAULT 0,
    age_20s INTEGER DEFAULT 0,
    age_30s INTEGER DEFAULT 0,
    age_40s INTEGER DEFAULT 0,
    age_50plus INTEGER DEFAULT 0
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_venue ON events(venue_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_pseudo ON events(pseudo_id);
CREATE INDEX IF NOT EXISTS idx_daily_stats_venue ON daily_stats(venue_id);
CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);
