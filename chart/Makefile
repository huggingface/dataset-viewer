.PHONY: init
init:
	helm dependency update .

.PHONY: quality
quality:
	helm lint
	helm lint --values env/staging.yaml
	helm lint --values env/prod.yaml
