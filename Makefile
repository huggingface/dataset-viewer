# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27060
export PORT_ADMIN := 8181
export PORT_API := 8180
export PORT_ROWS := 8182
export PORT_SEARCH := 8183
export PORT_SSE_API := 8185
export PORT_WORKER := 8186
export PORT_WEBHOOK := 8187
export PORT_REVERSE_PROXY := 8100

export ADMIN_UVICORN_PORT := ${PORT_ADMIN}
export API_UVICORN_PORT := ${PORT_API}
export ROWS_UVICORN_PORT := ${PORT_ROWS}
export SEARCH_UVICORN_PORT := ${PORT_SEARCH}
export SSE_API_UVICORN_PORT := ${PORT_SSE_API}
export WORKER_UVICORN_PORT := ${PORT_WORKER}
export WEBHOOK_UVICORN_PORT := ${PORT_WEBHOOK}

export API_HF_JWT_PUBLIC_KEY_URL := https://hub-ci.huggingface.co/api/keys/jwt
export API_HF_JWT_ADDITIONAL_PUBLIC_KEYS :=

# environment variables per target
start: export COMPOSE_PROJECT_NAME := datasets-server
stop: export COMPOSE_PROJECT_NAME := datasets-server
dev-start: export COMPOSE_PROJECT_NAME := dev-datasets-server
dev-stop: export COMPOSE_PROJECT_NAME := dev-datasets-server

.PHONY: start
start:
	docker compose --env-file .env up -d --build --force-recreate --remove-orphans --renew-anon-volumes --wait --wait-timeout 20

.PHONY: stop
stop:
	docker compose down --remove-orphans --volumes

.PHONY: dev-start
dev-start:
	docker compose --env-file .env --env-file .env.debug up -d --build --force-recreate --remove-orphans --renew-anon-volumes --wait --wait-timeout 20

.PHONY: dev-stop
dev-stop: stop

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e

# for install, quality checks and tests of every job, lib, service or worker, see the Makefile in the corresponding folder

.PHONY: install
install:
	$(MAKE) -C libs/libcommon install
	$(MAKE) -C libs/libapi install
	$(MAKE) -C jobs/cache_maintenance install
	$(MAKE) -C jobs/mongodb_migration install
	$(MAKE) -C services/admin install
	$(MAKE) -C services/api install
	$(MAKE) -C services/rows install
	$(MAKE) -C services/search install
	$(MAKE) -C services/sse-api install
	$(MAKE) -C services/worker install
	$(MAKE) -C services/webhook install
	$(MAKE) -C e2e install
