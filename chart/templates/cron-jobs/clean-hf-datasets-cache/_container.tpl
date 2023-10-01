# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerCleanHfDatasetsCache" -}}
- name: "{{ include "name" . }}-clean-hf-datasets-cache"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountHfDatasetsCacheRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.cleanHfDatasetsCache.resources | nindent 4 }}
  env:
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.cleanHfDatasetsCache.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.cleanHfDatasetsCache.log.level | quote }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ .Values.hfDatasetsCache.cacheDirectory | quote }}
  - name: DATASETS_BASED_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.hfDatasetsCache.expiredTimeIntervalSeconds | quote }}
{{- end -}}
