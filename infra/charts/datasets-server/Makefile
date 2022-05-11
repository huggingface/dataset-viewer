CHART_NAME := datasets-server
K8S_NAMESPACE := hub

.PHONY: init
init:
	helm dependency update .

.PHONY: uninstall
uninstall:
	helm uninstall $(CHART_NAME)-$(ENV) -n $(K8S_NAMESPACE)

.PHONY: diff
diff:
	helm diff upgrade --install $(CHART_NAME)-$(ENV) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: upgrade
upgrade:
	helm upgrade --install $(CHART_NAME)-$(ENV) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: diff-dev
diff-dev:
	@make diff ENV=dev

.PHONY: upgrade-dev
upgrade-dev:
	@make upgrade ENV=dev

.PHONY: quality
quality:
	helm lint
