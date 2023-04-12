# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMetricsCollector" -}}
- name: "{{ include "name" . }}-metrics-collector"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envLog" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envMetric" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.metricsCollector.action | quote }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.metricsCollector.resources | nindent 4 }}
{{- end -}}
