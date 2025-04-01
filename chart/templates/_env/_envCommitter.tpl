# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCommitter" -}}
- name: COMMITTER_HF_TOKEN
  {{- if .Values.secrets.appParquetConverterHfToken.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.appParquetConverterHfToken.secretName | default (include "datasetsServer.infisical.secretName" $) | quote }}
      key: PARQUET_CONVERTER_HF_TOKEN
      optional: false
  {{- else }}
  value: {{ .Values.secrets.appParquetConverterHfToken.value }}
  {{- end }}
{{- end -}}
