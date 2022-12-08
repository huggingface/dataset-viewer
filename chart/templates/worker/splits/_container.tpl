# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSplits" -}}
- name: "{{ include "name" . }}-worker-splits"
  image: {{ .Values.dockerImage.workers.datasets_based }}
  imagePullPolicy: IfNotPresent
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/splits"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  {{ include "envDatasetsWorker" . | nindent 2 }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.splits.queue.maxJobsPerNamespace | quote }}
  volumeMounts:
  {{ include "volumeMountCacheDatasets" . | nindent 2 }}
  {{ include "volumeMountCacheModules" . | nindent 2 }}
  {{ include "volumeMountCacheNumba" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.splits.resources | nindent 4 }}
{{- end -}}
