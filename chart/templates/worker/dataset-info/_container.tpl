# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerDatasetInfo" -}}
- name: "{{ include "name" . }}-worker-dataset-info"
  image: {{ .Values.dockerImage.workers.datasets_based }}
  imagePullPolicy: {{ .Values.docker.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/dataset-info"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.datasetInfo.queue.maxJobsPerNamespace | quote }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.datasetInfo.resources | nindent 4 }}
{{- end -}}
