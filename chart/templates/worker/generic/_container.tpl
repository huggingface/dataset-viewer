# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerGeneric" -}}
- name: "{{ include "name" . }}-worker-generic"
  image: {{ include "services.worker.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/generic/datasets" .Values.cacheDirectory | quote }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.genericWorker.queue.maxJobsPerNamespace | quote }}
  - name: FIRST_ROWS_MAX_BYTES
    value: {{ .Values.firstRows.maxBytes | quote }}
  - name: FIRST_ROWS_MAX_NUMBER
    value: {{ .Values.firstRows.maxNumber | quote }}
  - name: FIRST_ROWS_MIN_CELL_BYTES
    value: {{ .Values.firstRows.minCellBytes | quote }}
  - name: FIRST_ROWS_MIN_NUMBER
    value: {{ .Values.firstRows.minNumber| quote }}
  - name: FIRST_ROWS_COLUMNS_MAX_NUMBER
    value: {{ .Values.firstRows.columnsMaxNumber| quote }}
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
  {{ include "volumeMountAssetsRW" . | nindent 2 }}
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.genericWorker.resources | nindent 4 }}
{{- end -}}
