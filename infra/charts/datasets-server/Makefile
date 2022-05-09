K8S_NAMESPACE := hub

.PHONY: init
init:
	helm dependency update .

.PHONY: uninstall
uninstall:
	helm uninstall $(ENV) -n $(K8S_NAMESPACE)

.PHONY: diff
diff:
	helm diff upgrade --install $(ENV) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: upgrade
upgrade:
	helm upgrade --install $(ENV) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: diff-dev
diff-dev:
	@make diff ENV=dev

.PHONY: upgrade-dev
upgrade-dev:
	@make upgrade ENV=dev

.PHONY: quality
quality:
	helm lint
