# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountParquetMetadataRW" -}}
- mountPath: {{ .Values.parquetMetadata.storageDirectory | quote }}
  mountPropagation: None
  name: data
  subPath: "{{ include "parquetMetadata.subpath" . }}"
  readOnly: false
{{- end -}}
