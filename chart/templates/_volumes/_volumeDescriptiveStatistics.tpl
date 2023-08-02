# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeDescriptiveStatistics" -}}
- name: volume-descriptive-statistics
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.descriptiveStatistics.existingClaim | default (include "name" .) }}
{{- end -}}
