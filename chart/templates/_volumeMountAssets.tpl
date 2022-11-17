# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountAssets" -}}
- mountPath: {{ .Values.cache.assetsDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "assets.subpath" . }}"
  readOnly: true
{{- end -}}
