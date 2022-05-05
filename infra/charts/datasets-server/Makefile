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

.PHONY: install-ephemeral
install-ephemeral:
	@make upgrade ENV=ephemeral

.PHONY: diff-ephemeral
diff-ephemeral:
	@make diff ENV=ephemeral

.PHONY: upgrade-ephemeral
upgrade-ephemeral:
	@make upgrade ENV=ephemeral

.PHONY: quality
quality:
	helm lint
