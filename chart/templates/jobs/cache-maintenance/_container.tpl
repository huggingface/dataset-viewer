# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerCacheMaintenance" -}}
- name: "{{ include "name" . }}-cache-maintenance"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envMetrics" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cacheMaintenance.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cacheMaintenance.log.level | quote }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.cacheMaintenance.resources | nindent 4 }}
{{- end -}}
