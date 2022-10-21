# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSplits" -}}
- name: "{{ include "name" . }}-worker-splits"
  image: {{ .Values.dockerImage.workers.splits }}
  imagePullPolicy: IfNotPresent
  env:
  - name: CACHE_ASSETS_DIRECTORY
    value: {{ .Values.cache.assetsDirectory | quote }}
  - name: CACHE_MONGO_DATABASE
    value: {{ .Values.cache.mongoDatabase | quote }}
  - name: CACHE_MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  - name: QUEUE_MAX_JOBS_PER_DATASET
    value: {{ .Values.queue.maxJobsPerDataset | quote }}
  - name: QUEUE_MAX_LOAD_PCT
    value: {{ .Values.queue.maxLoadPct | quote }}
  - name: QUEUE_MAX_MEMORY_PCT
    value: {{ .Values.queue.maxMemoryPct | quote }}
  - name: QUEUE_MONGO_DATABASE
    value: {{ .Values.queue.mongoDatabase | quote }}
  - name: QUEUE_MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  - name: QUEUE_WORKER_SLEEP_SECONDS
    value: {{ .Values.queue.sleepSeconds | quote }}
  - name: COMMON_ASSETS_BASE_URL
    value: "{{ include "assets.baseUrl" . }}"
  - name: COMMON_HF_ENDPOINT
    value: {{ .Values.common.hfEndpoint | quote }}
  - name: COMMON_HF_TOKEN
    value: {{ .Values.secrets.hfToken | quote }}
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
  volumeMounts:
  - mountPath: {{ .Values.hfDatasetsCache | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  resources:
    {{ toYaml .Values.splits.resources | nindent 4 }}
{{- end -}}
