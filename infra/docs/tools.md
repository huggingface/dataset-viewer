## Tools

To work on the infrastructure, various CLI tools are required or recommended.

### aws

`aws` is the CLI for the AWS services. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html to install it.

You will mainly use:

- `aws configure sso` to login. See the [ECR section below](#amazon-elastic-container-registry-ecr).
- `aws ecr` to list, pull, push the docker images to the ECR repository. See the [ECR section below](#amazon-elastic-container-registry-ecr).
- `aws eks` to inspect the Kubernetes clusters, and setup `kubectl`. See [the "clusters" section in kube/ README](./kube/README.md#clusters).

### kubectl

`kubectl` is the Kubernetes CLI. See https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/ to install it on Linux.

Once installed, you can [alias](https://www.google.com/search?q=persist+alias+linux) it to `k` in your bash/zsh profile so that:

```
$ alias | grep kubectl
k=kubectl
```

To use it, you have to configure it to use a specific cluster using `aws eks`. See [the "clusters" section in kube/ README](./kube/README.md#clusters).
