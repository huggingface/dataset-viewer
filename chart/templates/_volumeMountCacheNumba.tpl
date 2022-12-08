# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountCacheNumba" -}}
- mountPath: {{ .Values.numbaCacheDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "cache.numba.subpath" . }}"
  readOnly: false
{{- end -}}
