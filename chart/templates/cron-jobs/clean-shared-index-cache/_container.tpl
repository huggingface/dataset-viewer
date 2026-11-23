# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

{{- define "containerCleanSharedIndexCache" -}}
- name: "{{ include "name" . }}-clean-shared-index-cache"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  {{- include "datasetsServer.csi.volumeMountBlock" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cleanSharedIndexCache.resources | nindent 4 }}
  env:
    {{ include "envLog" . | nindent 2 }}
    {{ include "envCache" . | nindent 2 }}
    {{ include "envQueue" . | nindent 2 }}
    {{ include "envSecrets" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: "clean-shared-index-cache"
  - name: SHARED_INDEX_CACHE_CLEANING_BUCKET
    value: {{ .Values.search.sharedIndexCache.bucket | quote }}
  - name: SHARED_INDEX_CACHE_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.cleanSharedIndexCache.expiredTimeIntervalSeconds | quote }}
{{- end -}}
