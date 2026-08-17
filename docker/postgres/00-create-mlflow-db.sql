-- Separate database for MLflow's own tracking-server schema (managed by its
-- Alembic migrations, see docker/mlflow) — kept out of the `mdal` database so
-- MLflow's schema never collides with the application tables in init.sql.
CREATE DATABASE mlflow;
