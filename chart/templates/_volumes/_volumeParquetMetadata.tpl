# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeParquetMetadata" -}}
- name: volume-parquet-metadata
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.parquetMetadata.existingClaim | default (include "name" .) }}
{{- end -}}
