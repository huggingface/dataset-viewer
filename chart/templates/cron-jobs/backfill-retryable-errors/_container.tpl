# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

{{- define "containerBackfillRetryableErrors" -}}
- name: "{{ include "name" . }}-backfill-retryable-errors"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.backfillRetryableErrors.resources | nindent 4 }}
  env:
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
    {{ include "envS3" . | nindent 2 }}
    {{ include "envAssets" . | nindent 2 }}
    {{ include "envCachedAssets" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.backfillRetryableErrors.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.backfillRetryableErrors.log.level | quote }}
{{- end -}}
