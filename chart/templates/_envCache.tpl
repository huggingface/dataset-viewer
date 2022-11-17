# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCache" -}}
- name: CACHE_ASSETS_DIRECTORY
  value: {{ .Values.cache.assetsDirectory | quote }}
- name: CACHE_MONGO_DATABASE
  value: {{ .Values.cache.mongoDatabase | quote }}
- name: CACHE_MONGO_URL
  {{- if .Values.secrets.mongoUrl.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.mongoUrl.secretName | quote }}
      key: MONGO_URL
      optional: false
  {{- else }}
    {{- if .Values.mongodb.enabled }}
  value: mongodb://{{.Release.Name}}-mongodb
    {{- else }}
  value: {{ .Values.secrets.mongoUrl.value }}
    {{- end }}
  {{- end }}
{{- end -}}
