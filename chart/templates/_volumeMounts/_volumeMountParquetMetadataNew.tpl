# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "volumeMountParquetMetadataNewRO" -}}
- mountPath: {{ .Values.parquetMetadata.storageDirectoryNew | quote }}
  mountPropagation: None
  name: volume-parquet-metadata
  subPath: "{{ include "parquetMetadata.subpath" . }}"
  readOnly: true
{{- end -}}

{{- define "volumeMountParquetMetadataNewRW" -}}
- mountPath: {{ .Values.parquetMetadata.storageDirectoryNew | quote }}
  mountPropagation: None
  name: volume-parquet-metadata
  subPath: "{{ include "parquetMetadata.subpath" . }}"
  readOnly: false
{{- end -}}
