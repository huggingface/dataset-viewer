# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountCacheModules" -}}
- mountPath: {{ .Values.hfModulesCache | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "cache.modules.subpath" . }}"
  readOnly: false
{{- end -}}
