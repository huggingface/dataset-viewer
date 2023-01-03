# build with
#   docker build -t some_tag_worker -f Dockerfile ../..
FROM python:3.9.15-slim

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/app

RUN pip install dagit

# Copy your code and workspace to /opt/dagster/app
COPY services/orchestrator/src/orchestrator/repo.py services/orchestrator/src/orchestrator/workspace.yaml /opt/dagster/app/

ENV DAGSTER_HOME=/opt/dagster/dagster_home/

# Copy dagster instance YAML to $DAGSTER_HOME
COPY services/orchestrator/src/orchestrator/dagster.yaml /opt/dagster/dagster_home/

WORKDIR /opt/dagster/app

EXPOSE 3000

ENTRYPOINT ["dagit", "--host", "0.0.0.0", "--port", "3000", "--path-prefix", "/admin/orchestrator"]
