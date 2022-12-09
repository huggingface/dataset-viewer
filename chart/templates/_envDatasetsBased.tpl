# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envDatasetsBased" -}}
# the size should remain so small that we don't need to worry about putting it on an external storage
# note that the /tmp directory is not shared among the pods
- name: HF_MODULES_CACHE
value: "/tmp/modules-cache"
- name: NUMBA_CACHE_DIR
value: "/tmp/numba-cache"
{{- end -}}

