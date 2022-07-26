## Tools

To work on the infrastructure, various CLI tools are required or recommended.

### aws

`aws` is the CLI for the AWS services. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html to install it.

You will mainly use:

- `aws configure sso` to login. See [authentication.md](./authentication.md).
- `aws ecr` to list, pull, push the docker images to the ECR repository. See [docker.md](./docker.md).
- `aws eks` to inspect the Kubernetes clusters, and setup `kubectl`. See [kubernetes.md](./kubernetes.md#clusters).

### kubectl

`kubectl` is the Kubernetes CLI. See https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/ to install it on Linux.

To use it, you have to configure it to use a specific cluster using `aws eks`. See [the "clusters" section in kube/ README](./kubernetes.md#clusters).

Once installed, you can:

- add [autocompletion](https://kubernetes.io/docs/reference/kubectl/cheatsheet/#kubectl-autocomplete)
- create an [alias](https://www.google.com/search?q=persist+alias+linux) to `k`: `alias k="kubectl"`
- install [kubectx and kubens](https://github.com/ahmetb/kubectx) to switch easily between [contexts](./kubernetes.md#context) and [namespaces](./kubernetes.md#namespaces)
- install [fzf](https://github.com/junegunn/fzf) and [kube-fzf](https://github.com/thecasualcoder/kube-fzf): command-line fuzzy searching of Kubernetes Pods
- install [kubelens](https://github.com/kubelens/kubelens): web application to look at the objects

### helm

Helm is a package manager for Kubernetes, and installs a [chart](https://helm.sh/docs/topics/charts/) (all of the resource definitions necessary to run an application, tool, or service inside of a Kubernetes cluster) into Kubernetes.

See https://helm.sh/docs/intro/install/ to install the `helm` CLI.

Once installed, you can:

- add [autocompletion](https://helm.sh/docs/helm/helm_completion/#see-also)
- install [helm-diff](https://github.com/databus23/helm-diff): a helm plugin that shows a diff explaining what a helm upgrade would change.

### make

Install `make` to use the [Makefile](../charts/datasets-server/Makefile) to deploy to the Kubernetes cluster:

```
sudo apt install make
```
