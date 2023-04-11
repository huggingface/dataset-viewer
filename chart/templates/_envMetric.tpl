# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envMetric" -}}
- name: METRIC_MONGO_DATABASE
  value: {{ .Values.queue.mongoDatabase | quote }}
- name: METRIC_MONGO_URL
  {{ include "datasetServer.mongo.url" . | nindent 2 }}
{{- end -}}
