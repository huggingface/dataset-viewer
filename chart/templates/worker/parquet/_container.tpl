# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerParquet" -}}
- name: "{{ include "name" . }}-worker-parquet"
  image: {{ .Values.dockerImage.workers.parquet }}
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
    value: {{ .Values.parquet.queue.maxJobsPerNamespace | quote }}
  - name: PARQUET_COMMIT_MESSAGE
    value: {{ .Values.parquet.commitMessage | quote }}
  - name: PARQUET_SOURCE_REVISION
    value: {{ .Values.parquet.sourceRevision | quote }}
  - name: PARQUET_SUPPORTED_DATASETS
    value: {{ .Values.parquet.supportedDatasets | quote }}
  - name: PARQUET_TARGET_REVISION
    value: {{ .Values.parquet.targetRevision | quote }}
  - name: PARQUET_URL_TEMPLATE
    value: {{ .Values.parquet.urlTemplate | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRO" . | nindent 2 }}
  {{ include "volumeMountDatasetsCache" . | nindent 2 }}
  {{ include "volumeMountNumbaCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.parquet.resources | nindent 4 }}
{{- end -}}
