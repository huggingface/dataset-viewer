# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerWebhook" -}}
- name: "{{ include "name" . }}-webhook"
  image: {{ include "services.webhook.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envS3" . | nindent 2 }}
  {{ include "envCloudfront" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envHf" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envNumba" . | nindent 2 }}
  # storage
  {{ include "envAssets" . | nindent 2 }}
  {{ include "envCachedAssets" . | nindent 2 }}
  # service
  - name: API_MAX_AGE_LONG
    value: {{ .Values.webhook.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.webhook.maxAgeShort | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.webhook.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.webhook.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.webhook.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.webhook.uvicornPort | quote }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.webhook.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.webhook.uvicornPort }}
  ports:
  - containerPort: {{ .Values.webhook.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.webhook.resources | nindent 4 }}
{{- end -}}
