# Helm

We use [Helm](https://helm.sh/docs/intro/using_helm/) to describe the Kubernetes resources of the `datasets-server` application (as a "Chart"), and deploy it to the Kubernetes cluster.

The [templates/](../charts/datasets-server/templates) directory contains a list of templates of Kubernetes resources configurations.

The [values.yaml](../charts/datasets-server/values.yaml) file contains a list of configuration values that are used in the templates to replace the placeholders. It can be overridden in all the `helm` command by the `--values` option (see how it is used in the [`Makefile`](../charts/datasets-server/Makefile)).
