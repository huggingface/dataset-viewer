# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerAdmin" -}}
- name: "{{ include "name" . }}-admin"
  image: {{ include "services.admin.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envMetrics" . | nindent 2 }}
  # storage
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  - name: DUCKDB_INDEX_CACHE_DIRECTORY
    value: {{ .Values.duckDBIndex.cacheDirectory | quote }}
  - name: DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY
    value: {{ .Values.descriptiveStatistics.cacheDirectory | quote }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ .Values.hfDatasetsCache.cacheDirectory | quote }}
  # service
  - name: ADMIN_HF_ORGANIZATION
    value: {{ .Values.admin.hfOrganization | quote }}
  - name: ADMIN_CACHE_REPORTS_NUM_RESULTS
    value: {{ .Values.admin.cacheReportsNumResults | quote }}
  - name: ADMIN_CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS
    value: {{ .Values.admin.cacheReportsWithContentNumResults | quote }}
  - name: ADMIN_HF_TIMEOUT_SECONDS
    value: {{ .Values.admin.hfTimeoutSeconds | quote }}
  - name: ADMIN_HF_WHOAMI_PATH
    value: {{ .Values.admin.hfWhoamiPath | quote }}
  - name: ADMIN_MAX_AGE
    value: {{ .Values.admin.maxAge | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.admin.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: ADMIN_UVICORN_HOSTNAME
    value: {{ .Values.admin.uvicornHostname | quote }}
  - name: ADMIN_UVICORN_NUM_WORKERS
    value: {{ .Values.admin.uvicornNumWorkers | quote }}
  - name: ADMIN_UVICORN_PORT
    value: {{ .Values.admin.uvicornPort | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRO" . | nindent 2 }}
  {{ include "volumeMountDescriptiveStatisticsRO" . | nindent 2 }}
  {{ include "volumeMountDuckDBIndexRO" . | nindent 2 }}
  {{ include "volumeMountHfDatasetsCacheRO" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRO" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.admin.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.admin.uvicornPort }}
  ports:
  - containerPort: {{ .Values.admin.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.admin.resources | nindent 4 }}
{{- end -}}
