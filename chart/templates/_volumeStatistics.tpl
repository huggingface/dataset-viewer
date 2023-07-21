# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeStatistics" -}}
- name: statistics-data
  persistentVolumeClaim:
    claimName: {{ .Values.statisticsPersistence.existingClaim | default (include "name" .) }}
{{- end -}}