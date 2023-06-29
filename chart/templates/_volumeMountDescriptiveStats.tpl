# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDescriptiveStatsRW" -}}
- mountPath: {{ .Values.descriptiveStats.storageDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "descriptiveStats.subpath" . }}"
  readOnly: false
{{- end -}}