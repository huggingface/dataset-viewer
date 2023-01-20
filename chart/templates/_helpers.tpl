# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{/*
Expand the name of the chart.
*/}}
{{- define "name" -}}
{{- ((list $.Release.Name .Chart.Name) | join "-") | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Expand the name of the release.
*/}}
{{- define "release" -}}
{{- default .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Docker image management
*/}}
{{- define "datasetsServer.images.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $separator := ":" -}}
{{- $termination := .imageRoot.tag | toString -}}
{{- if .global }}
    {{- if and .global.imageRegistry .imageRoot.useGlobalRegistry }}
     {{- $registryName = .global.imageRegistry -}}
    {{- end -}}
{{- end -}}
{{- if .imageRoot.digest }}
    {{- $separator = "@" -}}
    {{- $termination = .imageRoot.digest | toString -}}
{{- end -}}
{{- printf "%s/%s%s%s" $registryName $repositoryName $separator $termination -}}
{{- end -}}

{{- define "common.images.pullSecrets" -}}
  {{- $pullSecrets := list }}

  {{- if .global }}
    {{- range .global.imagePullSecrets -}}
      {{- $pullSecrets = append $pullSecrets . -}}
    {{- end -}}
  {{- end -}}

  {{- range .images -}}
    {{- range .pullSecrets -}}
      {{- $pullSecrets = append $pullSecrets . -}}
    {{- end -}}
  {{- end -}}

  {{- if (not (empty $pullSecrets)) }}
imagePullSecrets:
    {{- range $pullSecrets }}
  - name: {{ . }}
    {{- end }}
  {{- end }}
{{- end -}}

{{- define "reverseproxy.image" -}}
{{ include "datasetsServer.images.image" (dict "imageRoot" .Values.images.reverseProxy "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "jobs.mongodbMigration.image" -}}
{{ include "datasetsServer.images.image" (dict "imageRoot" .Values.images.jobs.mongodbMigration "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.admin.image" -}}
{{ include "datasetsServer.images.image" (dict "imageRoot" .Values.images.services.admin "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.api.image" -}}
{{ include "datasetsServer.images.image" (dict "imageRoot" .Values.images.services.api "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "workers.datasetsBased.image" -}}
{{ include "datasetsServer.images.image" (dict "imageRoot" .Values.images.workers.datasetsBased "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "image.imagePullSecrets" -}}
{{ include "common.images.pullSecrets" (dict "images" (list .Values.images) "global" .Values.global.huggingface) }}
{{- end -}}





{{/*
Selector labels
*/}}
{{- define "datasetServer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "datasetServer.labels" -}}
helm.sh/chart: {{ include "chart" . }}
{{ include "datasetServer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
release: {{ $.Release.Name | quote }}
heritage: {{ $.Release.Service | quote }}
chart: "{{ include "name" . }}"
{{- end }}

{{- define "labels.reverseProxy" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-reverse-proxy"
{{- end -}}

{{- define "labels.storageAdmin" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-storage-admin"
{{- end -}}

{{- define "labels.mongodbMigration" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-mongodb-migration"
{{- end -}}

{{- define "labels.admin" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-admin"
{{- end -}}

{{- define "labels.api" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-api"
{{- end -}}

{{- define "labels.splits" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-splits"
{{- end -}}

{{- define "labels.firstRows" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-first-rows"
{{- end -}}

{{- define "labels.parquetAndDatasetInfo" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-parquet-and-dataset-info"
{{- end -}}

{{- define "labels.parquet" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-parquet"
{{- end -}}

{{- define "labels.datasetInfo" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-dataset-info"
{{- end -}}

{{- define "labels.sizes" -}}
{{ include "datasetServer.labels" . }}
app: "{{ include "release" . }}-worker-sizes"
{{- end -}}

{{/*
Return the api ingress anotation
*/}}
{{- define "datasetsServer.ingress.annotations" -}}
{{ .Values.ingress.annotations | toYaml }}
{{- end -}}

{{/*
Datasets Server base url
*/}}
{{- define "datasetsServer.ingress.hostname" -}}
{{ .Values.global.huggingface.ingress.subdomains.datasetsServer }}.{{ .Values.global.huggingface.ingress.domain }}
{{- end }}

{{/*
The assets base URL
*/}}
{{- define "assets.baseUrl" -}}
{{- printf "https://%s/assets" (include "datasetsServer.ingress.hostname" .) }}
{{- end }}

{{/*
The assets/ subpath in the NFS
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same dir
*/}}
{{- define "assets.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "assets" }}
{{- end }}

{{/*
The datasets library will use this directory as a cache
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same dir
*/}}
{{- define "cache.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "cache" }}
{{- end }}

{{/*
The URL to access the mongodb instance created if mongodb.enable is true
It's named using the Release name
*/}}
{{- define "mongodb.url" -}}
{{- printf "mongodb://%s-mongodb" .Release.Name }}
{{- end }}

{{/*
The URL to access the admin service from another container
See https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/#a-aaaa-records
*/}}
{{- define "admin.url" -}}
{{- printf "http://%s-admin.%s.svc.cluster.local:80" ( include "name" . ) ( .Release.Namespace ) }}
{{- end }}

{{/*
The URL to access the API service from another container
See https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/#a-aaaa-records
*/}}
{{- define "api.url" -}}
{{- printf "http://%s-api.%s.svc.cluster.local:80" ( include "name" . ) ( .Release.Namespace ) }}
{{- end }}

{{/*
Return true if cert-manager required annotations for TLS signed
certificates are set in the Ingress annotations
Ref: https://cert-manager.io/docs/usage/ingress/#supported-annotations
Usage:
{{ include "common.ingress.certManagerRequest" ( dict "annotations" .Values.path.to.the.ingress.annotations ) }}
*/}}
{{- define "common.ingress.certManagerRequest" -}}
{{ if or (hasKey .annotations "cert-manager.io/cluster-issuer") (hasKey .annotations "cert-manager.io/issuer") (hasKey .annotations "kubernetes.io/tls-acme") }}
    {{- true -}}
{{- end -}}
{{- end -}}

{{/*
Return the HUB url
*/}}
{{- define "datasetsServer.hub.url" -}}
{{- if ne "" .Values.common.hfEndpoint }}
{{ .Values.common.hfEndpoint | quote }}
{{- else -}}
{{- $hubName := ((list $.Release.Name "hub") | join "-") | trunc 63 | trimSuffix "-" -}}
http://{{ $hubName }}
{{- end -}}
{{- end -}}