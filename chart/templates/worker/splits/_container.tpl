# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSplits" -}}
- name: "{{ include "name" . }}-worker-splits"
  image: {{ .Values.dockerImage.workers.splits }}
  imagePullPolicy: IfNotPresent
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  - name: HF_ENDPOINT # see https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411
    value: {{ .Values.common.hfEndpoint | quote }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.splits.queue.maxJobsPerNamespace | quote }}
  - name: QUEUE_MAX_LOAD_PCT
    value: {{ .Values.queue.maxLoadPct | quote }}
  - name: QUEUE_MAX_MEMORY_PCT
    value: {{ .Values.queue.maxMemoryPct | quote }}
  - name: QUEUE_WORKER_SLEEP_SECONDS
    value: {{ .Values.queue.sleepSeconds | quote }}
  - name: HF_DATASETS_CACHE
    value: {{ .Values.hfDatasetsCache | quote }}
  - name: HF_MODULES_CACHE
    value: "/tmp/modules-cache"
    # the size should remain so small that we don't need to worry about putting it on an external storage
    # see https://github.com/huggingface/datasets-server/issues/248
  - name: NUMBA_CACHE_DIR
    value: {{ .Values.numbaCacheDirectory | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRO" . | nindent 2 }}
  {{ include "volumeMountDatasetsCache" . | nindent 2 }}
  {{ include "volumeMountNumbaCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.splits.resources | nindent 4 }}
{{- end -}}
