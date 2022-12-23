# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envWorkerLoop" -}}
- name: WORKER_LOOP_MAX_DISK_USAGE_PCT
  value: {{ .Values.workerLoop.maxDiskUsagePct | quote }}
- name: WORKER_LOOP_MAX_LOAD_PCT
  value: {{ .Values.workerLoop.maxLoadPct | quote }}
- name: WORKER_LOOP_MAX_MEMORY_PCT
  value: {{ .Values.workerLoop.maxMemoryPct | quote }}
- name: WORKER_LOOP_SLEEP_SECONDS
  value: {{ .Values.workerLoop.sleepSeconds | quote }}
{{- end -}}
