# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountHfDatasetsCacheRW" -}}
- mountPath: {{ .Values.hfDatasetsCache.cacheDirectory | quote }}
  mountPropagation: None
  name: volume-hf-datasets-cache
  subPath: "{{ include "hfDatasetsCache.subpath" . }}"
  readOnly: false
{{- end -}}
