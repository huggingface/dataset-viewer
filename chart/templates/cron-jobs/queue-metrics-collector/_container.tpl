# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerQueueMetricsCollector" -}}
- name: "{{ include "name" . }}-queue-metrics-collector"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.queueMetricsCollector.resources | nindent 4 }}
  env:
    {{ include "envLog" . | nindent 2 }}
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.queueMetricsCollector.action | quote }}
{{- end -}}
