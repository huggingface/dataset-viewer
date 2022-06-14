.PHONY: down
down:	
	docker-compose -f $(DOCKER_COMPOSE) down -v --remove-orphans

.PHONY: up
up:	
	docker-compose -f $(DOCKER_COMPOSE) up -d
