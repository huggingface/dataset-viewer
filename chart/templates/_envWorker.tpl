# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envWorker" -}}
- name: WORKER_CONTENT_MAX_BYTES
  value: {{ .Values.worker.contentMaxBytes | quote}}
  # WORKER_ENDPOINT is not defined here, it's hard-coded in the template
- name: WORKER_MAX_DISK_USAGE_PCT
  value: {{ .Values.worker.maxDiskUsagePct | quote }}
- name: WORKER_MAX_LOAD_PCT
  value: {{ .Values.worker.maxLoadPct | quote }}
- name: WORKER_MAX_MEMORY_PCT
  value: {{ .Values.worker.maxMemoryPct | quote }}
- name: WORKER_SLEEP_SECONDS
  value: {{ .Values.worker.sleepSeconds | quote }}
- name: WORKER_STORAGE_PATHS
  value: {{ .Values.assets.storageDirectory | quote }}
  # ^ note: for datasets_based workers, the datasets cache is automatically added, so no need to add it here
{{- end -}}
