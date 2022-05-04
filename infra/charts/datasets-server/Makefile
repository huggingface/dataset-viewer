RELEASE_NAME := datasets-server
K8S_NAMESPACE := hub

.PHONY: install
diff:
	helm install $(RELEASE_NAME) . --values env/$(ENV).yaml --values env/infra-$(ENV).json -n $(K8S_NAMESPACE)

.PHONY: diff
diff:
	helm diff upgrade $(RELEASE_NAME) . --values env/$(ENV).yaml --values env/infra-$(ENV).json -n $(K8S_NAMESPACE)

.PHONY: upgrade
upgrade:
	helm upgrade $(RELEASE_NAME) . --values env/$(ENV).yaml --values env/infra-$(ENV).json -n $(K8S_NAMESPACE)

.PHONY: install-ephemeral
install-ephemeral:
	@make install ENV=ephemeral

.PHONY: diff-ephemeral
diff-ephemeral:
	@make diff ENV=ephemeral

.PHONY: upgrade-ephemeral
upgrade-ephemeral:
	@make upgrade ENV=ephemeral
