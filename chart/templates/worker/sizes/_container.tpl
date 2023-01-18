# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSizes" -}}
- name: "{{ include "name" . }}-worker-sizes"
  image: {{ .Values.dockerImage.workers.datasets_based }}
  imagePullPolicy: {{ .Values.docker.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/sizes"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.sizes.queue.maxJobsPerNamespace | quote }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.sizes.resources | nindent 4 }}
{{- end -}}
