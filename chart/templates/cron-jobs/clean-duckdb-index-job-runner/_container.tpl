# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerCleanDuckdbIndexJobRunner" -}}
- name: "{{ include "name" . }}-clean-duckdb-index-job-runner"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cleanDuckdbIndexJobRunner.resources | nindent 4 }}
  env:
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cleanDuckdbIndexJobRunner.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cleanDuckdbIndexJobRunner.log.level | quote }}
  - name: DIRECTORY_CLEANING_CACHE_DIRECTORY
    value: {{ .Values.duckDBIndex.cacheDirectory | quote }}
  - name: DIRECTORY_CLEANING_SUBFOLDER_PATTERN
    value: {{ .Values.cleanDuckdbIndexJobRunner.subfolderPattern | quote }}
  - name: DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.cleanDuckdbIndexJobRunner.expiredTimeIntervalSeconds | quote }}
{{- end -}}
