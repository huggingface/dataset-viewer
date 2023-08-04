# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerDeleteIndexes" -}}
- name: "{{ include "name" . }}-delete-indexes"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.deleteIndexes.resources | nindent 4 }}
  env:
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
    {{ include "envMetrics" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.deleteIndexes.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.deleteIndexes.log.level | quote }}
  - name: DUCKDB_INDEX_CACHE_DIRECTORY
    value: {{ .Values.duckDBIndex.cacheDirectory}}
  - name: DUCKDB_INDEX_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.duckDBIndex.expiredTimeIntervalSeconds}}
{{- end -}}
