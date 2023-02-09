# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerParquetAndDatasetInfo" -}}
- name: "{{ include "name" . }}-worker-parquet-and-dataset-info"
  image: {{ include "workers.datasetsBased.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/parquet-and-dataset-info"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/parquet-and-dataset-info/datasets" .Values.cacheDirectory | quote }}
  - name: DATASETS_BASED_CONTENT_MAX_BYTES
    value: {{ .Values.datasetsBased.contentMaxBytes | quote}}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.parquetAndDatasetInfo.queue.maxJobsPerNamespace | quote }}
  - name: PARQUET_AND_DATASET_INFO_BLOCKED_DATASETS
    value: {{ .Values.parquetAndDatasetInfo.blockedDatasets | quote }}
  - name: PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE
    value: {{ .Values.parquetAndDatasetInfo.commitMessage | quote }}
  - name: PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN
    {{- if .Values.secrets.userHfToken.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.userHfToken.secretName | quote }}
        key: HF_TOKEN
        optional: false
    {{- else }}
    value: {{ .Values.secrets.userHfToken.value }}
    {{- end }}
  - name: PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE
    value: {{ .Values.parquetAndDatasetInfo.maxDatasetSize | quote }}
  - name: PARQUET_AND_DATASET_INFO_SOURCE_REVISION
    value: {{ .Values.parquetAndDatasetInfo.sourceRevision | quote }}
  - name: PARQUET_AND_DATASET_INFO_SUPPORTED_DATASETS
    value: {{ .Values.parquetAndDatasetInfo.supportedDatasets | quote }}
  - name: PARQUET_AND_DATASET_INFO_TARGET_REVISION
    value: {{ .Values.parquetAndDatasetInfo.targetRevision | quote }}
  - name: PARQUET_AND_DATASET_INFO_URL_TEMPLATE
    value: {{ .Values.parquetAndDatasetInfo.urlTemplate | quote }}
  volumeMounts:
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.parquetAndDatasetInfo.resources | nindent 4 }}
{{- end -}}
