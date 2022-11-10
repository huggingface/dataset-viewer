# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerFirstRows" -}}
- name: "{{ include "name" . }}-worker-first-rows"
  image: {{ .Values.dockerImage.workers.firstRows }}
  imagePullPolicy: IfNotPresent
  env:
  - name: CACHE_ASSETS_DIRECTORY
    value: {{ .Values.cache.assetsDirectory | quote }}
  - name: CACHE_MONGO_DATABASE
    value: {{ .Values.cache.mongoDatabase | quote }}
  - name: CACHE_MONGO_URL
  {{- if .Values.secrets.mongoUrl.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl.secretName | quote }}
        key: MONGO_URL
        optional: false
  {{- else }}
    {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
    {{- else }}
    value: {{ .Values.secrets.mongoUrl.value }}
    {{- end }}
  {{- end }}
  - name: QUEUE_MONGO_DATABASE
    value: {{ .Values.queue.mongoDatabase | quote }}
  - name: QUEUE_MONGO_URL
  {{- if .Values.secrets.mongoUrl.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl.secretName | quote }}
        key: MONGO_URL
        optional: false
  {{- else }}
    {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
    {{- else }}
    value: {{ .Values.secrets.mongoUrl.value }}
    {{- end }}
  {{- end }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.firstRows.queue.maxJobsPerNamespace | quote }}
  - name: QUEUE_MAX_LOAD_PCT
    value: {{ .Values.queue.maxLoadPct | quote }}
  - name: QUEUE_MAX_MEMORY_PCT
    value: {{ .Values.queue.maxMemoryPct | quote }}
  - name: QUEUE_WORKER_SLEEP_SECONDS
    value: {{ .Values.queue.sleepSeconds | quote }}
  - name: COMMON_ASSETS_BASE_URL
    value: "{{ include "assets.baseUrl" . }}"
  - name: COMMON_HF_ENDPOINT
    value: {{ .Values.common.hfEndpoint | quote }}
  - name: COMMON_HF_TOKEN
  {{- if .Values.secrets.token.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.token.secretName | quote }}
        key: HF_TOKEN
        optional: false
  {{- else }}
    value: {{ .Values.secrets.token.value }}
  {{- end }}
  - name: COMMON_LOG_LEVEL
    value: {{ .Values.common.logLevel | quote }}
  - name: HF_DATASETS_CACHE
    value: {{ .Values.hfDatasetsCache | quote }}
  - name: HF_MODULES_CACHE
    value: "/tmp/modules-cache"
    # the size should remain so small that we don't need to worry about putting it on an external storage
    # see https://github.com/huggingface/datasets-server/issues/248
  - name: NUMBA_CACHE_DIR
    value: {{ .Values.numbaCacheDirectory | quote }}
  - name: FIRST_ROWS_FALLBACK_MAX_DATASET_SIZE
    value: {{ .Values.firstRows.fallbackMaxDatasetSize | quote }}
  - name: FIRST_ROWS_MAX_BYTES
    value: {{ .Values.firstRows.maxBytes | quote }}
  - name: FIRST_ROWS_MAX_NUMBER
    value: {{ .Values.firstRows.maxNumber | quote }}
  - name: FIRST_ROWS_MIN_CELL_BYTES
    value: {{ .Values.firstRows.minCellBytes | quote }}
  - name: FIRST_ROWS_MIN_NUMBER
    value: {{ .Values.firstRows.minNumber| quote }}
  volumeMounts:
  - mountPath: {{ .Values.cache.assetsDirectory | quote }}
    mountPropagation: None
    name: data
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.hfDatasetsCache | quote }}
    mountPropagation: None
    name: data
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: data
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  resources:
    {{ toYaml .Values.firstRows.resources | nindent 4 }}
{{- end -}}
