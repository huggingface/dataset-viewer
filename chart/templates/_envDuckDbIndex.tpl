# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envDuckDBIndex" -}}
- name: DUCKDB_INDEX_STORAGE_DIRECTORY
  value: {{ .Values.duckDBIndex.storageDirectory | quote }}
{{- end -}}
