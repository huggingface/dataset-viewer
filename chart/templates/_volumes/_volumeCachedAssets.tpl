# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeCachedAssets" -}}
- name: volume-cached-assets
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.cachedAssets.existingClaim | default (include "name" .) }}
{{- end -}}
