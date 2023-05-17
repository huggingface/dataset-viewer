# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envParquetMetadata" -}}
- name: PARQUET_METADATA_STORAGE_DIRECTORY
  value: {{ .Values.parquetMetadata.storageDirectory | quote }}
{{- end -}}
