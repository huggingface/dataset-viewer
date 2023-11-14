# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerDeleteObsoleteCache" -}}
- name: "{{ include "name" . }}-delete-obsolete-cache"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.deleteObsoleteCache.resources | nindent 4 }}
  env:
    {{ include "envAssets" . | nindent 2 }}
    {{ include "envCachedAssets" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envS3" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.deleteObsoleteCache.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.deleteObsoleteCache.log.level | quote }}
{{- end -}}
