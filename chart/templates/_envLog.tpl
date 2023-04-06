# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envLog" -}}
- name: LOG_LEVEL
  value: {{ .Values.log.level | quote }}
{{- end -}}
