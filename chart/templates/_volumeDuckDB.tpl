# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeDuckDB" -}}
- name: duckdb-data
  persistentVolumeClaim:
    claimName: {{ .Values.duckdbPersistence.existingClaim | default (include "name" .) }}
{{- end -}}
