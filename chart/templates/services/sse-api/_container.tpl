# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerSseApi" -}}
- name: "{{ include "name" . }}-sse-api"
  image: {{ include "services.sseApi.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envHf" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envNumba" . | nindent 2 }}
  # service
  - name: API_MAX_AGE_LONG
    value: {{ .Values.sseApi.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.sseApi.maxAgeShort | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.sseApi.prometheusMultiprocDirectory | quote }}
  # /hub-cache
  - name: HUB_CACHE_BASE_URL
    value: "https://{{ include "datasetsServer.ingress.hostname" . }}"
  - name: HUB_CACHE_NUM_RESULTS_PER_PAGE
    value: {{ .Values.sseApi.hubCacheNumResultsPerPage | quote }}
  # uvicorn
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.sseApi.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.sseApi.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.sseApi.uvicornPort | quote }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.sseApi.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.sseApi.uvicornPort }}
  ports:
  - containerPort: {{ .Values.sseApi.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.sseApi.resources | nindent 4 }}
{{- end -}}
