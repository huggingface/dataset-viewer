# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeHfDatasetsCache" -}}
- name: volume-hf-datasets-cache
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.hfDatasetsCache.existingClaim | default (include "name" .) }}
{{- end -}}
