# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWorkerParquet" -}}
- name: "{{ include "name" . }}-worker-parquet"
  image: {{ .Values.dockerImage.workers.datasets_based }}
  imagePullPolicy: IfNotPresent
  env:
  - name: DATASETS_BASED_ENDPOINT
    value: "/parquet"
    # ^ hard-coded
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envWorkerLoop" . | nindent 2 }}
  {{ include "envDatasetsBased" . | nindent 2 }}
  - name: DATASETS_BASED_HF_DATASETS_CACHE
    value: {{ printf "%s/parquet/datasets" .Values.cacheDirectory | quote }}
  - name: QUEUE_MAX_JOBS_PER_NAMESPACE
    # value: {{ .Values.queue.maxJobsPerNamespace | quote }}
    # overridden
    value: {{ .Values.parquet.queue.maxJobsPerNamespace | quote }}
  - name: PARQUET_BLOCKED_DATASETS
    value: {{ .Values.parquet.blockedDatasets | quote }}
  - name: PARQUET_COMMIT_MESSAGE
    value: {{ .Values.parquet.commitMessage | quote }}
  - name: PARQUET_COMMITTER_HF_TOKEN
    {{- if .Values.secrets.userHfToken.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.userHfToken.secretName | quote }}
        key: HF_TOKEN
        optional: false
    {{- else }}
    value: {{ .Values.secrets.userHfToken.value }}
    {{- end }}
  - name: PARQUET_MAX_DATASET_SIZE
    value: {{ .Values.parquet.maxDatasetSize | quote }}
  - name: PARQUET_SOURCE_REVISION
    value: {{ .Values.parquet.sourceRevision | quote }}
  - name: PARQUET_SUPPORTED_DATASETS
    value: {{ .Values.parquet.supportedDatasets | quote }}
  - name: PARQUET_TARGET_REVISION
    value: {{ .Values.parquet.targetRevision | quote }}
  - name: PARQUET_URL_TEMPLATE
    value: {{ .Values.parquet.urlTemplate | quote }}
  volumeMounts:
  {{ include "volumeMountCache" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.parquet.resources | nindent 4 }}
{{- end -}}
