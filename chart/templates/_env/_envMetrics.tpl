# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envMetrics" -}}
- name: METRICS_MONGO_DATABASE
  value: {{ .Values.metrics.mongoDatabase | quote }}
- name: METRICS_MONGO_URL
  {{ include "datasetServer.mongo.url" . | nindent 2 }}
{{- end -}}
