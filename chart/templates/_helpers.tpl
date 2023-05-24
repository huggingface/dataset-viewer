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
{{- define "reverseproxy.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.reverseProxy "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "jobs.mongodbMigration.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.jobs.mongodbMigration "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "jobs.cacheMaintenance.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.jobs.cacheMaintenance "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.admin.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.admin "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.api.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.api "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.worker.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.worker "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "image.imagePullSecrets" -}}
{{- include "hf.common.images.renderPullSecrets" (dict "images" (list .Values.images) "context" $) -}}
{{- end -}}


{{/*
Common labels
*/}}
{{- define "labels.reverseProxy" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-reverse-proxy"
{{- end -}}

{{- define "labels.storageAdmin" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-storage-admin"
{{- end -}}

{{- define "labels.mongodbMigration" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-mongodb-migration"
{{- end -}}

{{- define "labels.cacheMaintenance" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-cache-maintenance"
{{- end -}}

{{- define "labels.metricsCollector" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-metrics-collector"
{{- end -}}

{{- define "labels.backfill" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-backfill"
{{- end -}}

{{- define "labels.admin" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-admin"
{{- end -}}

{{- define "labels.api" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-api"
{{- end -}}

{{- define "labels.worker" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-worker"
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
Return the ingress scheme
*/}}
{{- define "datasetsServer.ingress.scheme" -}}
{{- if .Values.global.huggingface.ingress.ssl -}}
https://
{{- else -}}
http://
{{- end -}}
{{- end -}}

{{/*
The assets base URL
*/}}
{{- define "assets.baseUrl" -}}
{{- printf "%s%s/assets" (include "datasetsServer.ingress.scheme" .) (include "datasetsServer.ingress.hostname" .) }}
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
The cached-assets base URL
*/}}
{{- define "cachedAssets.baseUrl" -}}
{{- printf "%s%s/cached-assets" (include "datasetsServer.ingress.scheme" .) (include "datasetsServer.ingress.hostname" .) }}
{{- end }}

{{/*
The cached-assets/ subpath in the NFS
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same dir
*/}}
{{- define "cachedAssets.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "cached-assets" }}
{{- end }}

{{/*
The parquet-metadata/ subpath in the NFS
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same dir
*/}}
{{- define "parquetMetadata.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "parquet-metadata" }}
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
Return the HUB url
*/}}
{{- define "datasetsServer.hub.url" -}}
{{- if ne "" .Values.common.hfEndpoint -}}
{{ .Values.common.hfEndpoint | quote }}
{{- else -}}
{{- $hubName := ((list $.Release.Name "hub") | join "-") | trunc 63 | trimSuffix "-" -}}
http://{{ $hubName }}
{{- end -}}
{{- end -}}
