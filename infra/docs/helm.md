# Helm

We use [Helm](https://helm.sh/docs/intro/using_helm/) to describe the Kubernetes resources of the `datasets-server` application (as a "Chart"), and deploy it to the Kubernetes cluster.

The [templates/](../charts/datasets-server/templates) directory contains a list of templates of Kubernetes resources configurations.

The [values.yaml](../charts/datasets-server/values.yaml) file contains a list of configuration values that are used in the templates to replace the placeholders. It can be overridden in all the `helm` command by the `--values` option (see how it is used in the [`Makefile`](../charts/datasets-server/Makefile)).

## Notes

An Helm Release is like an instance of the app, deployed on the Kubernetes cluster. You can have various Releases at the same time, for example moon-landing has one Release for each pull-request, allowing to test the hub on every branch. All is related to the instance name (we concatenate `.Chart.Name`, ie `datasets-server` and `.Release.Name`, or env, ie. `dev`, to get the instance name: `datasets-server-dev`), which must be used in the labels, so that the Kubernetes objects are related as expected in the same Release, and ignore the objects of the other Releases.

Note that Kubernetes is not [blue-green deployment](https://en.wikipedia.org/wiki/Blue-green_deployment) (blue-green: two environments, "blue" and "green", coexist, where one is active and the other is inactive, and upgrading the app consists in preparing the inactive one, then activating it instead of the other). Meanwhile, Kubernetes create the new pods (and delete the old ones) one by one, which can lead to a small period with some pods running the new version of the app, and other ones running the old version. This means that the application should take care of the retrocompatibility (writing to the database, to the filesystem).

### MongoDB

To deploy mongodb for a given release, we declare it as a dependency in the datasets-server [Chart.yaml](../charts/datasets-server/Chart.yaml). When deployed, it spawns a service named `datasets-server-mongodb` (the release name, followed by `-mongodb`). We can see it:

```
$ hubectl get service
datasets-server-mongodb                                     ClusterIP   172.20.84.193    <none>        27017/TCP      18h
...
```

Note that with the current configuration, the whole cluster has access to the mongodb service. It is not exposed to the exterior though, and thus we don't require authentication for now. If we want to access mongo from a local machine, we can forward the port:

```
$ kubectl port-forward datasets-server-mongodb-0 27017:27017
Forwarding from 127.0.0.1:27017 -> 27017
Forwarding from [::1]:27017 -> 27017
```
