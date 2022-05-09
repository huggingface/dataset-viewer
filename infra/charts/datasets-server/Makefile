RELEASE_NAME := datasets-server
K8S_NAMESPACE := hub

.PHONY: init
init:
	helm dependency update .

.PHONY: uninstall
uninstall:
	helm uninstall $(RELEASE_NAME) -n $(K8S_NAMESPACE)

.PHONY: diff
diff:
	helm diff upgrade --install $(RELEASE_NAME) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: upgrade
upgrade:
	helm upgrade --install $(RELEASE_NAME) . --values env/$(ENV).yaml -n $(K8S_NAMESPACE)

.PHONY: diff-dev
diff-dev:
	@make diff ENV=dev

.PHONY: upgrade-dev
upgrade-dev:
	@make upgrade ENV=dev

.PHONY: quality
quality:
	helm lint
