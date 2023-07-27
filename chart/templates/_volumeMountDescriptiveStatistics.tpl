# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountDescriptiveStatisticsRW" -}}
- mountPath: {{ .Values.descriptiveStatistics.storageDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "descriptiveStatistics.subpath" . }}"
  readOnly: false
{{- end -}}
