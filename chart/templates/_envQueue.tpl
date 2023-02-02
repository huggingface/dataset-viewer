# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envQueue" -}}
- name: QUEUE_MONGO_CONNECTION_TIMEOUT_MS
  value: {{ .Values.queue.mongoConnectionTimeoutMs | quote }}
- name: QUEUE_MONGO_DATABASE
  value: {{ .Values.queue.mongoDatabase | quote }}
- name: QUEUE_MONGO_URL
  {{ include "datasetServer.mongo.url" . | nindent 2 }}
{{- end -}}
