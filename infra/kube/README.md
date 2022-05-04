# Kubernetes

## Clusters

All the projects that form part of the Hub, such as `datasets-server`, are deployed on a common Kubernetes cluster on Amazon EKS (Elastic Kubernetes Service). Two clusters are used:

- `hub-prod` for the production
- `hub-ephemeral` for the ephemeral environments (pull requests)

### List the clusters on Amazon EKS

If you have a profile with the rights to list the clusters on Amazon EKS, you can see them using the web console: https://us-east-1.console.aws.amazon.com/eks/home?region=us-east-1#/clusters, or use the CLI [`aws eks`](https://docs.aws.amazon.com/cli/latest/reference/eks/index.html):

```
$ aws eks list-clusters --profile=hub-pu
{
    "clusters": [
        "hub-ephemeral",
        "hub-prod"
    ]
}
```

Note that listing the clusters is not allowed for the `EKS-HUB-Hub` profile of the `hub` user:

```
$ aws eks list-clusters --profile=hub

An error occurred (AccessDeniedException) when calling the ListClusters operation: User: arn:aws:sts::707930574880:assumed-role/AWSReservedSSO_EKS-HUB-Hub_3c94769b0752b7d7/sylvain.lesage@huggingface.co is not authorized to perform: eks:ListClusters on resource: arn:aws:eks:us-east-1:707930574880:cluster/*
```

### Use a cluster

Setup `kubectl` to use a cluster:

```
$ aws eks update-kubeconfig --name=hub-ephemeral --profile=hub
Updated context arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral in /home/slesage/.kube/config
```

See the details of a cluster using `aws eks`:

```
$ aws eks describe-cluster --profile=hub --name=hub-ephemeral
{
    "cluster": {
        "name": "hub-ephemeral",
        "arn": "arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral",
        "createdAt": "2022-04-09T16:47:27.432000+00:00",
        "version": "1.22",
        ...
    }
}
```

## Namespaces

Get the list of namespaces of the current cluster:

```
$ kubectl get ns
NAME                 STATUS   AGE
dataset-server       Active   26h
default              Active   24d
gitaly               Active   24d
hub                  Active   24d
kube-node-lease      Active   24d
kube-public          Active   24d
kube-system          Active   24d
repository-scanner   Active   9d
```
