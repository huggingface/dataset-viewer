# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeData" -}}
- name: data
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.existingClaim | default (include "name" .) }}
{{- end -}}
