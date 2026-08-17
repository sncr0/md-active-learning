FROM postgres:16-alpine

COPY 00-create-mlflow-db.sql /docker-entrypoint-initdb.d/00-create-mlflow-db.sql
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql
