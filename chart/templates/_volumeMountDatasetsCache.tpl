# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountDatasetsCache" -}}
- mountPath: {{ .Values.hfDatasetsCache | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "cache.datasets.subpath" . }}"
  readOnly: false
{{- end -}}
