# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeCache" -}}
- name: cache-data
  persistentVolumeClaim:
    claimName: {{ .Values.cachePersistence.existingClaim | default (include "name" .) }}
{{- end -}}
