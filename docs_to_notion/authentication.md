## AWS CLI profile

To work on the `datasets-server` infrastructure, you have to configure AWS to use the SSO account `hub` (see https://huggingface.awsapps.com/start#/) with the role `EKS-HUB-Tensorboard` (see also the [doc in Notion about AWS SSO](https://www.notion.so/huggingface2/Conventions-645d29ce0a01496bb07c67a06612aa98#ff642cd8e28a4107ae26cc6183ccdd01)):

```shell
$ aws configure sso
SSO start URL [None]: https://huggingface.awsapps.com/start#/
SSO Region [None]: us-east-1
There are 3 AWS accounts available to you. # <-- select "hub"
Using the account ID 707930574880
There are 3 roles available to you. # <-- select "EKS-HUB-Tensorboard"
Using the role name "EKS-HUB-Tensorboard"
CLI default client Region [None]:
CLI default output format [None]:
CLI profile name [EKS-HUB-Hub-707930574880]: tb

To use this profile, specify the profile name using --profile, as shown:

aws s3 ls --profile tb
```

In the docs, we assume the AWS CLI profile is called `tb`.

The profile `tb` is meant to:

- operate inside the two EKS clusters (`hub-prod` and `hub-ephemeral`):

  - setup the kube contexts:

    ```shell
    aws eks update-kubeconfig --name "hub-prod" --alias "hub-prod-with-tb" --region us-east-1 --profile=tb
    aws eks update-kubeconfig --name "hub-ephemeral" --alias "hub-ephemeral-with-tb" --region us-east-1 --profile=tb
    ```

  - install kubectx and kubens (see [tools.md](./tools.md))
  - ephemeral:

    ```shell
    kubectx hub-ephemeral-with-tb
    kubens datasets-server
    kubectl get pod
    ```

  - prod:

    ```shell
    kubectx hub-prod-with-tb
    kubens datasets-server
    kubectl get pod
    ```

- list, pull, push docker images from repositories of the ECR registry (`707930574880.dkr.ecr.us-east-1.amazonaws.com`):

  ```shell
  $ aws ecr get-login-password --region us-east-1 --profile=tb \
    | docker login --username AWS --password-stdin 707930574880.dkr.ecr.us-east-1.amazonaws.com
  ```

It is not meant to operate on AWS resources directly. The following command gives authentication error for example:

```shell
$ aws eks list-clusters --profile=tb
```
