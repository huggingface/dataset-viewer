# Infra

## Description

The cloud infrastructure for the datasets-server uses:

- Amazon ECR to store the docker images of the datasets-server services. See the [docker/ README](./docker/README.md#amazon-elastic-container-registry-ecr).
- Amazon EKS for the Kubernetes clusters. See the [kube/ README](./kube/README.md#clusters).

Before starting, ensure to:

- [install the tools](#tools)
- [setup the AWS CLI profile](#aws-cli-profile)

## Tools

To work on the infrastructure, various CLI tools are required.

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

## AWS CLI profile

To work on the `datasets-server` infrastructure, you have to configure AWS to use the SSO account `hub` (see https://huggingface.awsapps.com/start#/) with the role `EKS-HUB-Hub` (see also the [doc in Notion about AWS SSO](https://www.notion.so/huggingface2/Conventions-645d29ce0a01496bb07c67a06612aa98#ff642cd8e28a4107ae26cc6183ccdd01)):

```shell
$ aws configure sso
SSO start URL [None]: https://huggingface.awsapps.com/start#/
SSO Region [None]: us-east-1
There are 3 AWS accounts available to you. # <-- select "hub"
Using the account ID 707930574880
There are 3 roles available to you. # <-- select "EKS-HUB-Hub"
Using the role name "EKS-HUB-Hub"
CLI default client Region [None]: us-east-1
CLI default output format [None]:
CLI profile name [EKS-HUB-Hub-707930574880]: hub

To use this profile, specify the profile name using --profile, as shown:

aws s3 ls --profile hub
```

In the docs, we assume the AWS CLI profile is called `hub`.

**Note**: until all the rights of the `EKS-HUB-Hub` role are setup adequately, you can use another role (create another profile called `hub-pu` by using `HFPowerUserAccess` instead of `EKS-HUB-Hub` in `aws configure sso`). This role will be removed soon.

The profile `hub` is meant to:

- operate inside the two EKS clusters (`hub-prod` and `hub-ephemeral`):

  ```shell
  $ aws eks describe-cluster --profile=hub --name=hub-ephemeral
  $ aws eks update-kubeconfig --profile=hub --name=hub-ephemeral
  ```

- list, pull, push docker images from repositories of the ECR registry (`707930574880.dkr.ecr.us-east-1.amazonaws.com`):

  ```shell
  $ aws ecr get-login-password --region us-east-1 --profile=hub \
    | docker login --username AWS --password-stdin 707930574880.dkr.ecr.us-east-1.amazonaws.com
  ```

  **Note**: the `EKS-HUB-Hub` profile still misses this right. Until the infra team adds it, you can use the `hub-pu` profile.

It is not meant to operate on AWS resources directly. The following command gives authentication error for example:

```shell
$ aws eks list-clusters --profile=hub
```
