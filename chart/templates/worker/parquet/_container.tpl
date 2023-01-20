# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerParquet" -}}
- name: "{{ include "name" . }}-worker-parquet"
  image: {{ include "workers.datasetsBased.image" . }}
  imagePullPolicy: {{ .Values.image.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/parquet"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.parquet.queue.maxJobsPerNamespace | quote }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.parquet.resources | nindent 4 }}
{{- end -}}
