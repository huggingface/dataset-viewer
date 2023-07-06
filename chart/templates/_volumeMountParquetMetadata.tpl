# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountParquetMetadataRO" -}}
- mountPath: {{ .Values.parquetMetadata.storageDirectory | quote }}
  mountPropagation: None
  name: parquet-data
  subPath: "{{ include "parquetMetadata.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountParquetMetadataRW" -}}
- mountPath: {{ .Values.parquetMetadata.storageDirectory | quote }}
  mountPropagation: None
  name: parquet-data
  subPath: "{{ include "parquetMetadata.subpath" . }}"
  readOnly: false
{{- end -}}
