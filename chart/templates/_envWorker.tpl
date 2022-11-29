# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envWorker" -}}
- name: WORKER_MAX_LOAD_PCT
  value: {{ .Values.worker.maxLoadPct | quote }}
- name: WORKER_MAX_MEMORY_PCT
  value: {{ .Values.worker.maxMemoryPct | quote }}
- name: WORKER_WORKER_SLEEP_SECONDS
  value: {{ .Values.worker.sleepSeconds | quote }}
{{- end -}}
