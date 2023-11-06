# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeNfs" -}}
- name: volume-nfs
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.nfs.existingClaim | default (include "name" .) }}
{{- end -}}
