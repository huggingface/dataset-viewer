# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerSplits" -}}
- name: "{{ include "name" . }}-worker-splits"
  env:
  - name: DATASETS_REVISION
    value: {{ .Values.worker.splits.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.worker.splits.cacheDirectory }}/datasets"
  - name: HF_ENDPOINT
    value: {{ .Values.hfEndpoint | quote }}
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
    value: {{ .Values.worker.splits.logLevel | quote }}
  - name: MAX_JOBS_PER_DATASET
    value: {{ .Values.worker.splits.maxJobsPerDataset | quote }}
  - name: MAX_LOAD_PCT
    value: {{ .Values.worker.splits.maxLoadPct | quote }}
  - name: MAX_MEMORY_PCT
    value: {{ .Values.worker.splits.maxMemoryPct | quote }}
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
    value: {{ .Values.worker.splits.numbaCacheDirectory | quote }}
  - name: WORKER_SLEEP_SECONDS
    value: {{ .Values.worker.splits.workerleepSeconds | quote }}
  image: {{ .Values.dockerImage.worker.splits }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.worker.splits.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.splits.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  resources:
    {{ toYaml .Values.worker.splits.resources | nindent 4 }}
{{- end -}}
