# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDuckDBIndexRW" -}}
- mountPath: {{ .Values.duckDBIndex.storageDirectory | quote }}
  mountPropagation: None
  name: duckdb-data
  subPath: "{{ include "duckDBIndex.subpath" . }}"
  readOnly: false
{{- end -}}
