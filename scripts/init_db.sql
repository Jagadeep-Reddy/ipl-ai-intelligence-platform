-- scripts/init_db.sql
-- Runs automatically when the postgres Docker container starts fresh.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ipl_db TO ipl;
