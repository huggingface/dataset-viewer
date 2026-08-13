# Dataset viewer Helm chart

The dataset viewer Helm [chart](https://helm.sh/docs/topics/charts/) describes the Kubernetes resources of the dataset viewer application.

If you have access to the internal HF notion, see https://www.notion.so/huggingface2/Infrastructure-b4fd07f015e04a84a41ec6472c8a0ff5.

The cloud infrastructure for the dataset viewer uses:

- Docker Hub to store the docker images of the dataset viewer services.
- Amazon EKS for the Kubernetes clusters.

Note that this Helm chart is used to manage the deployment of the dataset viewer services to the cloud infrastructure (AWS) using Kubernetes. The infrastructure in itself is not created here, but in https://github.com/huggingface/infra/ using terraform. If you need to create or modify some resources, contact the infra team.

## Secrets

The secrets live in Infisical, and the chart supports two ways of getting them to the pods.

### Operator mode (`secrets.infisical.enabled`)

The Infisical operator syncs the whole secret path into a Kubernetes Secret, and every pod reads the
values it needs through `valueFrom.secretKeyRef` environment variables. The operator authenticates with
Universal Auth, using the client ID and secret stored in `secrets.infisical.operatorSecretName`.

This is the historical mode. It keeps working if Infisical is unreachable, at the cost of the secret
material sitting in etcd and in the environment of every pod.

### CSI mode (`secrets.infisical.csi.enabled`)

Each pod mounts only the secrets it needs as files in a read-only tmpfs volume, and reads them once at
startup into process memory. No Kubernetes Secret is created and nothing is injected in the
environment, so the values no longer show up in etcd, in `kubectl get secret`, in `/proc/<pid>/environ`,
in the environment inherited by subprocesses, or in whatever an error reporter captures.

The node's CSI provider is what talks to Infisical, so the pods hold no credential of their own. An
application-side compromise cannot ask for a different secret path or a different environment: it can
only read the files that were mounted for it.

`secrets.infisical.csi.workloads` lists what each workload mounts. That list is the blast radius of a
compromised pod, so it is meant to be read and audited rather than defaulted.

The two modes are mutually exclusive, and the chart refuses to render if both are enabled.

Prerequisites, none of which live in this repository:

- the Secrets Store CSI Driver and the Infisical provider installed on the cluster
- an Infisical machine identity the provider authenticates with, granted read access to the
  `secrets.infisical.project` and `secrets.infisical.env` scope. Its ID and the project ID are injected
  at deploy time along with `secrets.infisical.url`, none of them being stored here.

Setting `windowDuration` bounds how long after a container starts its secrets stay readable; the files
are served empty afterwards, which shrinks the exposure of an application-side compromise from the
pod's whole life to a few minutes around startup. It needs a provider that supports it, and rotation
enabled on the driver. Two consequences worth knowing:

- Applications retry their first read for a short while, because granting a window takes a few seconds
  to take effect. `libcommon.secrets` does this, waiting up to `SECRETS_TIMEOUT_SECONDS`.
- A volume is the unit of enforcement while a container is the unit of decision, so a tight window
  wants one consumer per volume.

Local development, the tests and the e2e suite leave both modes off, and the services keep reading
their usual environment variables.

## Deploy

To deploy, go to https://cd.internal.huggingface.tech/applications.
