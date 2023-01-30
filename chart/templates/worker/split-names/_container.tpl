# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSplitNames" -}}
- name: "{{ include "name" . }}-worker-split-names"
  image: {{ include "workers.datasetsBased.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/split-names"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/split-names/datasets" .Values.cacheDirectory | quote }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.splitNames.queue.maxJobsPerNamespace | quote }}
  volumeMounts:
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.splitNames.resources | nindent 4 }}
{{- end -}}
