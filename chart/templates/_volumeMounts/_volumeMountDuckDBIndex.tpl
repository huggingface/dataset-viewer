# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDuckDBIndexRW" -}}
- mountPath: {{ .Values.duckDBIndex.cacheDirectory | quote }}
  mountPropagation: None
  name: volume-duckdb-index
  subPath: "{{ include "duckDBIndex.subpath" . }}"
  readOnly: false
{{- end -}}
