# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorker" -}}
- name: "{{ include "name" . }}-worker"
  image: {{ include "services.worker.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  {{ include "envDuckDBIndex" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/%s/datasets" .Values.cacheDirectory .workerValues.deployName | quote }}
  - name: WORKER_JOB_TYPES_BLOCKED
    value: {{ .workerValues.workerJobTypesBlocked | quote }}
  - name: WORKER_JOB_TYPES_ONLY
    value: {{ .workerValues.workerJobTypesOnly | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountCache" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRW" . | nindent 2 }}
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .workerValues.resources | nindent 4 }}
{{- end -}}
