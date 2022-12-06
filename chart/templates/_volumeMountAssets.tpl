# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountAssetsRO" -}}
- mountPath: {{ .Values.firstRows.assetsDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "assets.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountAssetsRW" -}}
- mountPath: {{ .Values.firstRows.assetsDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "assets.subpath" . }}"
  readOnly: false
{{- end -}}
