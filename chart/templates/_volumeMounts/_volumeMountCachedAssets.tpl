# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "volumeMountCachedAssetsRO" -}}
- mountPath: {{ .Values.cachedAssets.storageDirectory | quote }}
  mountPropagation: None
  name: volume-nfs
  subPath: "{{ include "cachedAssets.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountCachedAssetsRW" -}}
- mountPath: {{ .Values.cachedAssets.storageDirectory | quote }}
  mountPropagation: None
  name: volume-nfs
  subPath: "{{ include "cachedAssets.subpath" . }}"
  readOnly: false
{{- end -}}
