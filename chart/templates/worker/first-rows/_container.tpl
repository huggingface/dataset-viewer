# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerFirstRows" -}}
- name: "{{ include "name" . }}-worker-first-rows"
  image: {{ include "workers.datasetsBased.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/first-rows"
    # ^ hard-coded
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  - name: WORKER_LOOP_STORAGE_PATHS
    value: {{ .Values.assets.storageDirectory | quote }}
    # ^ note: the datasets cache is automatically added, so no need to add it here
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/first-rows/datasets" .Values.cacheDirectory | quote }}
  - name: DATASETS_BASED_CONTENT_MAX_BYTES
    value: {{ .Values.datasetsBased.contentMaxBytes | quote}}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.firstRows.queue.maxJobsPerNamespace | quote }}
  - name: FIRST_ROWS_MAX_BYTES
    value: {{ .Values.firstRows.maxBytes | quote }}
  - name: FIRST_ROWS_MAX_NUMBER
    value: {{ .Values.firstRows.maxNumber | quote }}
  - name: FIRST_ROWS_MIN_CELL_BYTES
    value: {{ .Values.firstRows.minCellBytes | quote }}
  - name: FIRST_ROWS_MIN_NUMBER
    value: {{ .Values.firstRows.minNumber| quote }}
  - name: FIRST_ROWS_COLUMNS_MAX_NUMBER
    value: {{ .Values.firstRows.columnsMaxNumber| quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.firstRows.resources | nindent 4 }}
{{- end -}}
