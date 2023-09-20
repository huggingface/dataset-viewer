# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerCacheCleaner" -}}
- name: "{{ include "name" . }}-cache-cleaner"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountCachedAssetsRW" . | nindent 2 }} 
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cacheCleaner.resources | nindent 4 }}
  env:
    {{ include "envAssets" . | nindent 2 }}
    {{ include "envCache" . | nindent 2 }}
    {{ include "envCachedAssets" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cacheCleaner.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cacheCleaner.log.level | quote }}
{{- end -}}
