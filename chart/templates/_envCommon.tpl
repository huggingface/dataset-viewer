# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCommon" -}}
- name: COMMON_HF_ENDPOINT
  value: {{ include "datasetsServer.hub.url" . }}
- name: HF_ENDPOINT # see https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411
  value: {{ include "datasetsServer.hub.url" . }}
- name: COMMON_HF_TOKEN
  {{- if .Values.secrets.appHfToken.fromSecret }}
  valueFrom:
    secretKeyRef:
      {{- if eq .Values.secrets.appHfToken.secretName "" }}
      name: {{ .Release.Name }}-datasets-server-app-token
      {{- else }}
      name: {{ .Values.secrets.appHfToken.secretName | quote }}
      {{- end }}
      key: HF_TOKEN
      optional: false
  {{- else }}
  value: {{ .Values.secrets.appHfToken.value }}
  {{- end }}
- name: COMMON_LOG_LEVEL
  value: {{ .Values.common.logLevel | quote }}
{{- end -}}

{{- define "datasetServer.mongo.url" -}}
{{- if .Values.secrets.mongoUrl.fromSecret }}
valueFrom:
  secretKeyRef:
    name: {{ .Values.secrets.mongoUrl.secretName | quote }}
    key: MONGO_URL
    optional: false
{{- else }}
  {{- if .Values.mongodb.enabled }}
value: mongodb://{{.Release.Name}}-datasets-server-mongodb
  {{- else }}
value: {{ .Values.secrets.mongoUrl.value }}
  {{- end }}
{{- end }}
{{- end -}}