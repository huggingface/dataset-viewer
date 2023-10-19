# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerCleanStatsCache" -}}
- name: "{{ include "name" . }}-clean-stats-cache"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountDescriptiveStatisticsRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cleanStatsCache.resources | nindent 4 }}
  env:
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cleanStatsCache.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cleanStatsCache.log.level | quote }}
  - name: DIRECTORY_CLEANING_CACHE_DIRECTORY
    value: {{ .Values.descriptiveStatistics.cacheDirectory | quote }}
  - name: DIRECTORY_CLEANING_SUBFOLDER_PATTERN
    value: {{ .Values.cleanStatsCache.subfolderPattern | quote }}
  - name: DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.cleanStatsCache.expiredTimeIntervalSeconds | quote }}
{{- end -}}
