.PHONY: down
down:	
	docker compose -f $(DOCKER_COMPOSE) down --remove-orphans --volumes

.PHONY: up
up:	
	docker compose -f $(DOCKER_COMPOSE) up -d --build --force-recreate --remove-orphans --renew-anon-volumes
	echo "rs.initiate({_id:'rs0',version:1,members:[{_id:0,host:'localhost:$(MONGO_PORT)'}]})" | mongosh --port $(MONGO_PORT) --quiet
