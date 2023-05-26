# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envNumba" -}}
# the size should remain so small that we don't need to worry about putting it on an external storage
# note that the /tmp directory is not shared among the pods
# This is needed to use numba and packages that use numba like librosa
- name: NUMBA_CACHE_DIR
  value: "/tmp/numba-cache"
{{- end -}}

