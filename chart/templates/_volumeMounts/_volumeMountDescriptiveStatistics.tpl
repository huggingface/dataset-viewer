# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDescriptiveStatisticsRO" -}}
- mountPath: {{ .Values.descriptiveStatistics.cacheDirectory | quote }}
  mountPropagation: None
  name: volume-descriptive-statistics
  subPath: "{{ include "descriptiveStatistics.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountDescriptiveStatisticsRW" -}}
- mountPath: {{ .Values.descriptiveStatistics.cacheDirectory | quote }}
  mountPropagation: None
  name: volume-descriptive-statistics
  subPath: "{{ include "descriptiveStatistics.subpath" . }}"
  readOnly: false
{{- end -}}
