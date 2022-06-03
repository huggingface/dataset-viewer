CHART_NAME := datasets-server

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
	@make diff ENV=dev K8S_NAMESPACE=hub

.PHONY: uninstall-dev
uninstall-dev:
	@make uninstall ENV=dev K8S_NAMESPACE=hub

.PHONY: upgrade-dev
upgrade-dev:
	@make upgrade ENV=dev K8S_NAMESPACE=hub

.PHONY: diff-prod
diff-prod:
	@make diff ENV=prod K8S_NAMESPACE=datasets-server

.PHONY: uninstall-prod
uninstall-prod:
	@make uninstall ENV=prod K8S_NAMESPACE=datasets-server

.PHONY: upgrade-prod
upgrade-prod:
	@make upgrade ENV=prod K8S_NAMESPACE=datasets-server

.PHONY: quality
quality:
	helm lint
