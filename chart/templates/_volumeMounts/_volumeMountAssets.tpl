# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountAssetsRO" -}}
- mountPath: {{ .Values.assets.storageDirectory | quote }}
  mountPropagation: None
  name: volume-nfs
  subPath: "{{ include "assets.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountAssetsRW" -}}
- mountPath: {{ .Values.assets.storageDirectory | quote }}
  mountPropagation: None
  name: volume-nfs
  subPath: "{{ include "assets.subpath" . }}"
  readOnly: false
{{- end -}}
