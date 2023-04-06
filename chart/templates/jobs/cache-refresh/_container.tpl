# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerCacheRefresh" -}}
- name: "{{ include "name" . }}-cache-refresh"
  image: {{ include "jobs.cacheRefresh.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.cacheRefresh.resources | nindent 4 }}
{{- end -}}
