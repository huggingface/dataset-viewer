# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeParquet" -}}
- name: parquet-data
  persistentVolumeClaim:
    claimName: {{ .Values.parquetPersistence.existingClaim | default (include "name" .) }}
{{- end -}}
