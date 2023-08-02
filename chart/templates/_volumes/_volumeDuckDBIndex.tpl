# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeDuckDBIndex" -}}
- name: volume-duckdb-index
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.duckDBIndex.existingClaim | default (include "name" .) }}
{{- end -}}
