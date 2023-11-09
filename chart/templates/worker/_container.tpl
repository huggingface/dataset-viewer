# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorker" -}}
- name: "{{ include "name" . }}-worker"
  image: {{ include "services.worker.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envS3" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envNumba" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envWorker" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/%s/datasets" .Values.hfDatasetsCache.cacheDirectory .workerValues.deployName | quote }}
  - name: WORKER_DIFFICULTY_MAX
    value: {{ .workerValues.workerDifficultyMax | quote }}
  - name: WORKER_DIFFICULTY_MIN
    value: {{ .workerValues.workerDifficultyMin | quote }}
  - name: WORKER_JOB_TYPES_BLOCKED
    value: {{ .workerValues.workerJobTypesBlocked | quote }}
  - name: WORKER_JOB_TYPES_ONLY
    value: {{ .workerValues.workerJobTypesOnly | quote }}
  - name: ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY
    value: {{ .Values.rowsIndex.maxArrowDataInMemory | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .workerValues.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: WORKER_UVICORN_HOSTNAME
    value: {{ .workerValues.uvicornHostname | quote }}
  - name: WORKER_UVICORN_NUM_WORKERS
    value: {{ .workerValues.uvicornNumWorkers | quote }}
  - name: WORKER_UVICORN_PORT
    value: {{ .workerValues.uvicornPort | quote }}
  volumeMounts:
  {{ include "volumeMountDescriptiveStatisticsRW" . | nindent 2 }}
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  {{ include "volumeMountHfDatasetsCacheRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRW" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .workerValues.resources | nindent 4 }}
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .workerValues.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .workerValues.uvicornPort }}
  ports:
  - containerPort: {{ .workerValues.uvicornPort }}
    name: http
    protocol: TCP
{{- end -}}
