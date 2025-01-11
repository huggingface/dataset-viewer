export API_HF_JWT_PUBLIC_KEY_URL := https://hub-ci.huggingface.co/api/keys/jwt
export API_HF_JWT_ADDITIONAL_PUBLIC_KEYS :=

.PHONY: down
down:	
	docker compose -f $(DOCKER_COMPOSE) down --remove-orphans --volumes

.PHONY: up
up:	
	docker compose -f $(DOCKER_COMPOSE) up -d --build --force-recreate --remove-orphans --renew-anon-volumes --wait --wait-timeout 20
