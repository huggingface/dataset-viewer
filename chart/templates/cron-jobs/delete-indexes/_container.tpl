# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerDeleteIndexes" -}}
- name: "{{ include "name" . }}-delete-indexes"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
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
  - name: DUCKDB_INDEX_STORAGE_DIRECTORY
    value: {{ .Values.duckDBIndex.storageDirectory}}
{{- end -}}
