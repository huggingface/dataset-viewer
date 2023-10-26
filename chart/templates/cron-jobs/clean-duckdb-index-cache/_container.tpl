# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerCleanDuckdbIndexCache" -}}
- name: "{{ include "name" . }}-clean-duckdb-cache"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cleanDuckdbIndexCache.resources | nindent 4 }}
  env:
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cleanDuckdbIndexCache.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cleanDuckdbIndexCache.log.level | quote }}
  - name: DIRECTORY_CLEANING_CACHE_DIRECTORY
    value: {{ .Values.duckDBIndex.cacheDirectory | quote }}
  - name: DIRECTORY_CLEANING_SUBFOLDER_PATTERN
    value: {{ .Values.cleanDuckdbIndexCache.subfolderPattern | quote }}
  - name: DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.cleanDuckdbIndexCache.expiredTimeIntervalSeconds | quote }}
{{- end -}}
