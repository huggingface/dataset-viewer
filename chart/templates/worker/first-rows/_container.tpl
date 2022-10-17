# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerFirstRows" -}}
- name: "{{ include "name" . }}-worker-first-rows"
  env:
  - name: ASSETS_BASE_URL
    value: "{{ include "assets.baseUrl" . }}"
  - name: ASSETS_DIRECTORY
    value: {{ .Values.worker.firstRows.assetsDirectory | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.worker.firstRows.cacheDirectory }}/datasets"
  - name: HF_ENDPOINT
    value: {{ .Values.hfEndpoint | quote }}
  # note: HF_MODULES_CACHE is not set to a shared directory
  - name: HF_MODULES_CACHE
    value: "/tmp/modules-cache"
  # the size should remain so small that we don't need to worry about putting it on an external storage
  # see https://github.com/huggingface/datasets-server/issues/248
  - name: HF_TOKEN
    # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
    # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfToken | quote }}
        key: HF_TOKEN
        optional: false
  - name: LOG_LEVEL
    value: {{ .Values.worker.firstRows.logLevel | quote }}
  - name: MAX_JOBS_PER_DATASET
    value: {{ .Values.worker.firstRows.maxJobsPerDataset | quote }}
  - name: MAX_LOAD_PCT
    value: {{ .Values.worker.firstRows.maxLoadPct | quote }}
  - name: MAX_MEMORY_PCT
    value: {{ .Values.worker.firstRows.maxMemoryPct | quote }}
  - name: MAX_SIZE_FALLBACK
    value: {{ .Values.worker.firstRows.maxSizeFallback | quote }}
  - name: MIN_CELL_BYTES
    value: {{ .Values.worker.firstRows.minCellBytes | quote }}
  - name: MONGO_CACHE_DATABASE
    value: {{ .Values.mongodb.cacheDatabase | quote }}
  - name: MONGO_QUEUE_DATABASE
    value: {{ .Values.mongodb.queueDatabase | quote }}
  - name: MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  - name: NUMBA_CACHE_DIR
    value: {{ .Values.worker.firstRows.numbaCacheDirectory | quote }}
  - name: ROWS_MAX_BYTES
    value: {{ .Values.worker.firstRows.rowsMaxBytes | quote }}
  - name: ROWS_MAX_NUMBER
    value: {{ .Values.worker.firstRows.rowsMaxNumber | quote }}
  - name: ROWS_MIN_NUMBER
    value: {{ .Values.worker.firstRows.rowsMinNumber| quote }}
  - name: WORKER_SLEEP_SECONDS
    value: {{ .Values.worker.firstRows.workerleepSeconds | quote }}
  image: {{ .Values.dockerImage.worker.firstRows }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.worker.firstRows.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.firstRows.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.firstRows.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  resources:
    {{ toYaml .Values.worker.firstRows.resources | nindent 4 }}
{{- end -}}
