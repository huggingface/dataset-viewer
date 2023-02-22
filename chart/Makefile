K8S_NAMESPACE := datasets-server

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
	helm lint --values env/dev.yaml
	helm lint --values env/prod.yaml
