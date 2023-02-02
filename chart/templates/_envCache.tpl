# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCache" -}}
- name: CACHE_MONGO_CONNECTION_TIMEOUT_MS
  value: {{ .Values.cache.mongoConnectionTimeoutMs | quote }}
- name: CACHE_MONGO_DATABASE
  value: {{ .Values.cache.mongoDatabase | quote }}
- name: CACHE_MONGO_URL
  {{ include "datasetServer.mongo.url" . | nindent 2 }}
{{- end -}}
