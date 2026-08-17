FROM python:3.12-slim

# psycopg2-binary: MLflow's Postgres backend-store driver (separate from the
# mdal app, which uses psycopg v3 — this container only ever talks to the
# `mlflow` database, never mdal's tables).
RUN pip install --no-cache-dir "mlflow>=2.14" "psycopg2-binary>=2.9"

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
