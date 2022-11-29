# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envDatasetsWorker" -}}
- name: HF_DATASETS_CACHE
  value: {{ .Values.hfDatasetsCache | quote }}
- name: HF_MODULES_CACHE
  value: "/tmp/modules-cache"
  # the size should remain so small that we don't need to worry about putting it on an external storage
  # see https://github.com/huggingface/datasets-server/issues/248
- name: NUMBA_CACHE_DIR
  value: {{ .Values.numbaCacheDirectory | quote }}
{{- end -}}
