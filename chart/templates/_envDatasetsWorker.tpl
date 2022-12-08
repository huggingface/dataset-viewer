# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envDatasetsWorker" -}}
- name: HF_DATASETS_CACHE
  value: {{ .Values.hfDatasetsCache | quote }}
- name: HF_MODULES_CACHE
  value: {{ .Values.hfModulesCache | quote }}
- name: NUMBA_CACHE_DIR
  value: {{ .Values.numbaCacheDirectory | quote }}
{{- end -}}
