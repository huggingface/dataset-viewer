# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDescriptiveStatisticsRW" -}}
- mountPath: {{ .Values.descriptiveStatistics.cacheDirectory | quote }}
  mountPropagation: None
  name: statistics-data
  subPath: "{{ include "descriptiveStatistics.subpath" . }}"
  readOnly: false
{{- end -}}
