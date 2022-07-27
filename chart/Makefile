CHART_NAME := datasets-server
K8S_NAMESPACE := datasets-server

.PHONY: init
init:
	helm dependency update .

.PHONY: uninstall
uninstall:
	helm uninstall $(CHART_NAME)-$(ENV) -n $(K8S_NAMESPACE)

.PHONY: diff
diff:
	helm diff upgrade --install $(CHART_NAME)-$(ENV) . --values docker-images.yaml --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: upgrade
upgrade:
	helm upgrade --install $(CHART_NAME)-$(ENV) . --values docker-images.yaml --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: diff-dev
diff-dev:
	@make diff ENV=dev

.PHONY: uninstall-dev
uninstall-dev:
	@make uninstall ENV=dev

.PHONY: upgrade-dev
upgrade-dev:
	@make upgrade ENV=dev

.PHONY: diff-prod
diff-prod:
	@make diff ENV=prod

.PHONY: uninstall-prod
uninstall-prod:
	@make uninstall ENV=prod

.PHONY: upgrade-prod
upgrade-prod:
	@make upgrade ENV=prod

.PHONY: quality
quality:
	helm lint
