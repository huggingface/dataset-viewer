# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envDiscussions" -}}
- name: DISCUSSIONS_BOT_ASSOCIATED_USER_NAME
  value: {{ .Values.discussions.botAssociatedUserName | quote }}
- name: DISCUSSIONS_BOT_TOKEN
  {{- if .Values.secrets.appParquetConverterHfToken.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.appParquetConverterHfToken.secretName | quote }}
      key: PARQUET_CONVERTER_HF_TOKEN
      optional: false
  {{- else }}
  value: {{ .Values.secrets.appParquetConverterHfToken.value }}
  {{- end }}
  # ^ we use the same token (datasets-server-bot) for discussions and for uploading parquet files
- name: DISCUSSIONS_PARQUET_REVISION
  value: {{ .Values.parquetAndInfo.targetRevision | quote }}
{{- end -}}
