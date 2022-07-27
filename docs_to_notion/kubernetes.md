# Kubernetes

This directory contains object configuration files, following the [Declarative object configuration](https://kubernetes.io/docs/concepts/overview/working-with-objects/object-management/#declarative-object-configuration) method of deploying an application on Kubernetes.

This means that we should only use `kubectl diff` and `kubectl apply` to manage the state (and `kubectl get` to read the values), and never use `kubectl create` or `kubectl delete`.

## Cluster

All the projects that form part of the Hub, such as `datasets-server`, are deployed on a common Kubernetes cluster on Amazon EKS (Elastic Kubernetes Service). Two clusters are available:

- `hub-prod` for the production
- `hub-ephemeral` for the ephemeral environments (pull requests)

### List the clusters on Amazon EKS

If you have a profile with the rights to list the clusters on Amazon EKS, you can see them using the web console: https://us-east-1.console.aws.amazon.com/eks/home?region=us-east-1#/clusters, or use the CLI [`aws eks`](https://docs.aws.amazon.com/cli/latest/reference/eks/index.html):

```
$ aws eks list-clusters --profile=hub-pu
{
    "clusters": [
        "hub-ephemeral",
        "hub-preprod",
        "hub-prod"
    ]
}
```

Note that listing the clusters is not allowed for the `EKS-HUB-Tensorboard` role of the `hub` account:

```
$ aws eks list-clusters --profile=tb

An error occurred (AccessDeniedException) when calling the ListClusters operation: User: arn:aws:sts::707930574880:assumed-role/AWSReservedSSO_EKS-HUB-Tensorboard_855674a9053d4044/sylvain.lesage@huggingface.co is not authorized to perform: eks:ListClusters on resource: arn:aws:eks:eu-west-3:707930574880:cluster/*
```

We've had to use another role to do it: create another profile called `hub-pu` by using `HFPowerUserAccess` instead of `EKS-HUB-Hub` in `aws configure sso`. Beware: this role might be removed soon.

### Use a cluster

Setup `kubectl` to use a cluster:

- prod:
  ```
  $ aws eks update-kubeconfig --name "hub-prod" --alias "hub-prod-with-tb" --region us-east-1 --profile=tb
  Updated context hub-prod-with-tb in /home/slesage/.kube/config
  ```
- ephemeral:
  ```
  $ aws eks update-kubeconfig --name "hub-ephemeral" --alias "hub-ephemeral-with-tb" --region us-east-1 --profile=tb
  Updated context hub-ephemeral-with-tb in /home/slesage/.kube/config
  ```

## Kubernetes objects

The principal Kubernetes objects within a cluster are:

- [namespace](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/): mechanism for isolating groups of resources within a single cluster
- [node](https://kubernetes.io/docs/tutorials/kubernetes-basics/explore/explore-intro/): the virtual or physical machines grouped in a cluster, each of which runs multiple pods. Note that with the `EKS-HUB-Hub` role, we don't have access to the list of nodes
- [deployment](https://kubernetes.io/docs/tutorials/kubernetes-basics/deploy-app/deploy-intro/): the configuration sent to the control plane to deploy and manage a containerized application. It describes a desired state for a set of pods
- [pod](https://kubernetes.io/docs/concepts/workloads/pods/): the pods are where the containerized applications are running, once deployed.
- [service](https://kubernetes.io/docs/concepts/services-networking/service/): an abstraction to access containerized application through the network from outside the cluster (maps a port on the proxy to the pods that will respond)
- [ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/): a set of rules that define how a service is exposed to the outside (URL, load-balancing, TLS, etc.)
- [configmap](https://kubernetes.io/docs/concepts/configuration/configmap/): configuration data for pods to consume.
- [secret](https://kubernetes.io/docs/concepts/configuration/secret/): secret data (like configmap, but confidential)

To get the complete list of object types:

```
kubectl api-resources -o wide | less
```

To get some help about an object type, use `kubectl explain`:

```
$ kubectl explain pod

KIND:     Pod
VERSION:  v1

DESCRIPTION:
     Pod is a collection of containers that can run on a host. This resource is
     created by clients and scheduled onto hosts.

...
```

### Useful kubectl commands

Some useful commands:

- `kubectl api-resources`: list all the object types (resources)
- `kubectl get xxx`: get the list of objects of type `xxx`. See also the [tips section](#tips-with-kubectl-get)
- `kubectl explain xxx`: get a description of what the `xxx` object type is.
- `kubectl logs pod/yyy`: show the logs of the pod `yyy`
- `kubectl exec pod/yyy -it sh`: open a shell on the pod `yyy`. More here: https://kubernetes.io/docs/reference/kubectl/cheatsheet/#interacting-with-running-pods and here: https://kubernetes.io/docs/reference/kubectl/cheatsheet/#interacting-with-deployments-and-services
- `kubectl describe xxx/yyy`: show the details of the object `yyy` of type `xxx`. In particular, look at the `Events` section at the end, to debug what occurs to the object.
  ```
    Type     Reason     Age                    From     Message
    ----     ------     ----                   ----     -------
    Warning  Unhealthy  28m (x2730 over 17h)   kubelet  Readiness probe failed: dial tcp 10.12.43.223:80: connect: connection refused
    Normal   Pulled     8m1s (x301 over 17h)   kubelet  Container image "707930574880.dkr.ecr.us-east-1.amazonaws.com/hub-datasets-server-api:sha-59db084" already present on machine
    Warning  BackOff    3m3s (x3643 over 17h)  kubelet  Back-off restarting failed container
  ```
- `kubectl rollout restart deploy/yyy`: recreate the pods of the deploy `yyy`
- `kubectl scale --replicas=5 deploy/yyy`: change (up or down, 0 is also valid) the number of replicas of the deploy `yyy`

### Tips with kubectl get

The `-o` option of `kubectl get xxx`, where `xxx` is the object type (`namespace`, `pod`, `deploy`...), allows to [format the output](https://kubernetes.io/docs/reference/kubectl/cheatsheet/#formatting-output):

- without the option `-o`: a table with a basic list of attributes and one line per object
- `-o wide`: a table with an extended list of attributes and one line per object
- `-o json`: a JSON object with the complete list of the objects and their (nested) attributes. Pipe into [`fx`](https://github.com/antonmedv/fx), `less`, `grep` or [`jq`](https://stedolan.github.io/jq/) to explore or extract info.
- `-o yaml`: the same as JSON, but in YAML format

You can filter to get the info only for one object by adding its name as an argument, eg:

- list of namespaces:

  ```
  kubectl get namespace -o json
  ```

- only the `datasets-server` namespace:

  ```
  kubectl get namespace datasets-server -o json
  ```

You can also filter by [label](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/):

- get the namespace with the name `datasets-server` (not very interesting):

  ```
  kubectl get namespace -l "kubernetes.io/metadata.name"==datasets-server
  ```

- get the pods of the `datasets-server-prod-api` application (note that `app` is a custom label specified in the Helm templates):

  ```
  kubectl get pod -l app==datasets-server-prod-api --namespace datasets-server
  ```

Use the `-w` option if you want to "watch" the values in real time.

Also note that every object type can be written in singular or plural, and also possibly in a short name (see `kubectl api-resources`), eg the following are equivalent

```
kubectl get namespace
kubectl get namespaces
kubectl get ns
```

More here: https://kubernetes.io/docs/reference/kubectl/cheatsheet/#viewing-finding-resources

## Other tips

Make your containerized applications listen to `0.0.0.0`, not `localhost`.

## Namespaces

Get the list of [namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/) of the current cluster (`hub-ephemeral`)):

```
$ kubectl get namespace
NAME                 STATUS   AGE
dataset-server       Active   26h
...
```

## Context

Contexts are useful to set the default namespace, user and cluster we are working on (see https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/).

We can create a local context called `hub-prod-with-tb` as:

```
$ kubectl config set-context \
    --cluster=arn:aws:eks:us-east-1:707930574880:cluster/hub-prod \
    --user=arn:aws:eks:us-east-1:707930574880:cluster/hub-prod \
    --namespace=datasets-server \
    hub-prod-with-tb
Context "hub-prod-with-tb" created.
```

or

```
$ kubectl config set-context \
    --cluster=arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral \
    --user=arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral \
    --namespace=datasets-server \
    hub-ephemeral-with-tb
Context "hub-ephemeral-with-tb" created.
```

Another way, seen before, is to use:

```shell
aws eks update-kubeconfig --name "hub-prod" --alias "hub-prod-with-tb" --region us-east-1 --profile=tb
aws eks update-kubeconfig --name "hub-ephemeral" --alias "hub-ephemeral-with-tb" --region us-east-1 --profile=tb
```

We set it as the current context with:

```
$ kubectl config use-context hub-ephemeral-with-tb

Switched to context "hub-ephemeral-with-tb".
```

If we list the contexts, we see that it is selected:

```
$ kubectl config get-contexts
CURRENT   NAME                    CLUSTER                                                    AUTHINFO                                                   NAMESPACE
*         hub-ephemeral-with-tb   arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral   arn:aws:eks:us-east-1:707930574880:cluster/hub-ephemeral   datasets-server
          hub-prod-with-tb        arn:aws:eks:us-east-1:707930574880:cluster/hub-prod        arn:aws:eks:us-east-1:707930574880:cluster/hub-prod        datasets-server
```

Note that contexts are a help for the developer to get quickly in the correct configuration. It's not stored in the cluster.

You might be interested in the `kubectx` and `kubens` tools (see https://github.com/ahmetb/kubectx) if you want to switch more easily between namespaces and contexts.

## Secrets

The HF token must be set manually in a secret (see https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-config-file/).

First, convert the secret to base64:

```
# Ask the Hub administrators to get an HF App token
$ echo -n 'hf_app_xxxx' | base64
yyyyy
```

Then paste it inside a secret configuration:

```
$ vi secret.yaml
```

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: datasets-server-hf-token
type: Opaque
data:
  HF_TOKEN: yyyyy
```

Finally create the secret:

```
kubectl apply -f ./secret.yaml
```

Alternatively, we can generate the secret with:

```shell
kubectl create secret generic datasets-server-hf-token --from-literal=HF_TOKEN='hf_app_xxxx'
```
