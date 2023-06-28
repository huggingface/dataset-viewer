# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDescriptiveStats" -}}
- mountPath: {{ .Values.descriptiveStats.storageDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "duckDBIndex.subpath" . }}"
  readOnly: false
{{- end -}}