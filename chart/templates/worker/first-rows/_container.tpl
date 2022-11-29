# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerFirstRows" -}}
- name: "{{ include "name" . }}-worker-first-rows"
  image: {{ .Values.dockerImage.workers.firstRows }}
  imagePullPolicy: IfNotPresent
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  {{ include "envDatasetsWorker" . | nindent 2 }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.firstRows.queue.maxJobsPerNamespace | quote }}
  - name: FIRST_ROWS_FALLBACK_MAX_DATASET_SIZE
    value: {{ .Values.firstRows.fallbackMaxDatasetSize | quote }}
  - name: FIRST_ROWS_MAX_BYTES
    value: {{ .Values.firstRows.maxBytes | quote }}
  - name: FIRST_ROWS_MAX_NUMBER
    value: {{ .Values.firstRows.maxNumber | quote }}
  - name: FIRST_ROWS_MIN_CELL_BYTES
    value: {{ .Values.firstRows.minCellBytes | quote }}
  - name: FIRST_ROWS_MIN_NUMBER
    value: {{ .Values.firstRows.minNumber| quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountDatasetsCache" . | nindent 2 }}
  {{ include "volumeMountNumbaCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.firstRows.resources | nindent 4 }}
{{- end -}}
