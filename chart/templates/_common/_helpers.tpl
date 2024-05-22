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

{{- define "services.rows.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.rows "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.search.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.search "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.sseApi.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.sseApi "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.worker.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.worker "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "services.webhook.image" -}}
{{ include "hf.common.images.image" (dict "imageRoot" .Values.images.services.webhook "global" .Values.global.huggingface) }}
{{- end -}}

{{- define "image.imagePullSecrets" -}}
{{- include "hf.common.images.renderPullSecrets" (dict "images" (list .Values.images) "context" $) -}}
{{- end -}}


{{/*
Common labels
*/}}
{{- define "labels.mongodbMigration" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-mongodb-migration"
{{- end -}}

{{- define "labels.queueMetricsCollector" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-queue-metrics-collector"
{{- end -}}

{{- define "labels.cacheMetricsCollector" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-cache-metrics-collector"
{{- end -}}

{{- define "labels.backfill" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-backfill"
{{- end -}}

{{- define "labels.backfillRetryableErrors" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-backfill-retryable-errors"
{{- end -}}

{{- define "labels.postMessages" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-post-messages"
{{- end -}}

{{- define "labels.admin" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-admin"
{{- end -}}

{{- define "labels.api" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-api"
{{- end -}}

{{- define "labels.rows" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-rows"
{{- end -}}

{{- define "labels.search" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-search"
{{- end -}}

{{- define "labels.sseApi" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-sse-api"
{{- end -}}

{{- define "labels.worker" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-worker-{{ .workerValues.deployName }}"
{{- end -}}

{{- define "labels.webhook" -}}
{{ include "hf.labels.commons" . }}
app.kubernetes.io/component: "{{ include "name" . }}-webhook"
{{- end -}}

{{/*
The dataset viewer API base url
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
The cached-assets base URL
*/}}
{{- define "cachedAssets.baseUrl" -}}
{{- printf "%s%s/cached-assets" (include "datasetsServer.ingress.scheme" .) (include "datasetsServer.ingress.hostname" .) }}
{{- end }}

{{/*
The parquet-metadata/ subpath in the EFS
- in a subdirectory named as the chart (dataset-viewer/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same dir
*/}}
{{- define "parquetMetadata.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "parquet-metadata" }}
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

{{/*
Return the api ingress anotation
note: keep $instanceAnnotations in first position during the merge, to avoid override annotations in other pods
*/}}
{{- define "datasetsServer.instance.ingress.annotations" -}}
{{- $instanceAnnotations := .instance.ingress.annotations -}}
{{- $defaultAnnotations := .context.Values.ingress.annotations -}}
{{- $dict := merge $instanceAnnotations $defaultAnnotations -}}
{{- range $key, $value := $dict }}
{{ $key | quote }}: {{ $value | quote }}
{{- end }}
{{- end -}}
