# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envWorker" -}}
- name: WORKER_CONTENT_MAX_BYTES
  value: {{ .Values.worker.contentMaxBytes | quote}}
- name: WORKER_HEARTBEAT_INTERVAL_SECONDS
  value: {{ .Values.worker.heartbeatIntervalSeconds | quote}}
- name: WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
  value: {{ .Values.worker.killZombiesIntervalSeconds | quote}}
- name: WORKER_MAX_DISK_USAGE_PCT
  value: {{ .Values.worker.maxDiskUsagePct | quote }}
- name: WORKER_MAX_LOAD_PCT
  value: {{ .Values.worker.maxLoadPct | quote }}
- name: WORKER_MAX_MEMORY_PCT
  value: {{ .Values.worker.maxMemoryPct | quote }}
- name: WORKER_MAX_MISSING_HEARTBEATS
  value: {{ .Values.worker.maxMissingHeartbeats | quote }}
- name: WORKER_SLEEP_SECONDS
  value: {{ .Values.worker.sleepSeconds | quote }}
- name: WORKER_STATE_FILE_PATH
  value: "/tmp/worker_state.json"
  # ^the size should remain so small that we don't need to worry about putting it on an external storage
  # note that the /tmp directory is not shared among the pods
- name: WORKER_STORAGE_PATHS
  value: {{ .Values.assets.storageDirectory | quote }}
# specific to the /first-rows job runner
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
# specific to the /parquet-and-dataset-info job runner
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
{{- end -}}
