# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerSearch" -}}
- name: "{{ include "name" . }}-search"
  image: {{ include "services.search.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envCachedAssets" . | nindent 2 }}
  {{ include "envS3" . | nindent 2 }}
  {{ include "envCloudfront" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envSecrets" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envHf" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  - name: HF_HOME
    value: "/tmp/hf"
    # ^ensure the temporary files are created in /tmp, which is writable
  # service
  - name: API_MAX_AGE_LONG
    value: {{ .Values.search.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.search.maxAgeShort | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.search.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.search.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.search.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.search.uvicornPort | quote }}
  # duckdb
  - name: DUCKDB_INDEX_TARGET_REVISION
    value: {{ .Values.duckDBIndex.targetRevision | quote }}
  - name: DUCKDB_INDEX_CACHE_DIRECTORY
    value: {{ .Values.duckDBIndex.cacheDirectory | quote }}
  - name: DUCKDB_INDEX_CACHE_CLEAN_CACHE_PROBA
    value: {{ .Values.search.cleanCacheProba | quote }}
  - name: DUCKDB_INDEX_CACHE_EXPIRED_TIME_INTERVAL_SECONDS
    value: {{ .Values.search.expiredTimeIntervalSeconds | quote }}
  - name: DUCKDB_INDEX_EXTENSIONS_DIRECTORY
    value: "/tmp/duckdb-extensions"
  {{- if .Values.search.sharedIndexCache.bucket }}
  - name: DUCKDB_INDEX_SHARED_CACHE_DIRECTORY
    value: {{ .Values.search.sharedIndexCache.mountPath | quote }}
  {{- end }}
  volumeMounts:
    {{ include "datasetsServer.csi.volumeMount" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRW" . | nindent 2 }}
  # An emptyDir survives container restarts, so a liveness kill no longer drops every built index
  - mountPath: {{ .Values.duckDBIndex.cacheDirectory | quote }}
    name: duckdb-index-cache
  {{- if .Values.search.sharedIndexCache.bucket }}
  - mountPath: {{ .Values.search.sharedIndexCache.mountPath | quote }}
    name: duckdb-index-shared
  {{- if and .Values.secrets.infisical.csi.enabled .Values.search.sharedIndexCache.infisicalSecretKey }}
  # The kubelet only mounts the volumes a container uses.
  - mountPath: "/mnt/duckdb-index-shared-token"
    name: duckdb-index-shared-token
    readOnly: true
  {{- end }}
  {{- end }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.search.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    timeoutSeconds: {{ .Values.search.livenessProbeTimeoutSeconds }}
    httpGet:
      path: /healthcheck
      port: {{ .Values.search.uvicornPort }}
  ports:
  - containerPort: {{ .Values.search.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.search.resources | nindent 4 }}
{{- end -}}
