# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountCache" -}}
- mountPath: {{ .Values.cacheDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "cache.subpath" . }}"
  readOnly: false
{{- end -}}
