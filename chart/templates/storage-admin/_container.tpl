# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerStorageAdmin" -}}
- name: "{{ include "name" . }}-storage-admin"
  image: {{ include "services.storageAdmin.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountDescriptiveStatisticsRW" . | nindent 2 }}
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  {{ include "volumeMountHfDatasetsCacheRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataNewRW" . | nindent 2 }}
  - mountPath: /volumes/descriptive-statistics
    mountPropagation: None
    name: volume-descriptive-statistics
    readOnly: false
  - mountPath: /volumes/duckdb-index
    mountPropagation: None
    name: volume-duckdb-index
    readOnly: false
  - mountPath: /volumes/hf-datasets-cache
    mountPropagation: None
    name: volume-hf-datasets-cache
    readOnly: false
  - mountPath: /volumes/nfs
    mountPropagation: None
    name: volume-nfs
    readOnly: false
  - mountPath: /volumes/parquet-metadata
    mountPropagation: None
    name: volume-parquet-metadata
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
  resources: {{ toYaml .Values.storageAdmin.resources | nindent 4 }}
{{- end -}}
