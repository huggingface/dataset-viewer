# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMetricsCollector" -}}
- name: "{{ include "name" . }}-metrics-collector"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  securityContext: {{ include "securityContext" . | nindent 4 }}
  resources: {{ toYaml .Values.metricsCollector.resources | nindent 4 }}
  env:
    {{ include "envLog" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
    {{ include "envMetrics" . | nindent 2 }}
{{- end -}}
