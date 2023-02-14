# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerParquet" -}}
- name: "{{ include "name" . }}-worker-parquet"
  image: {{ include "services.worker.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  - name: WORKER_ONLY_JOB_TYPES
    value: "/parquet"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/parquet/datasets" .Values.cacheDirectory | quote }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.parquet.queue.maxJobsPerNamespace | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.parquet.resources | nindent 4 }}
{{- end -}}
