# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerCacheMonitor" -}}
- name: "{{ include "name" . }}-cache-monitor"
  image: {{ include "jobs.cacheMonitor.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envLog" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.cacheMonitor.resources | nindent 4 }}
{{- end -}}
