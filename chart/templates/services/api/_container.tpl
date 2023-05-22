# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerApi" -}}
- name: "{{ include "name" . }}-api"
  image: {{ include "services.api.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCachedAssets" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  # service
  - name: API_HF_AUTH_PATH
    value: {{ .Values.api.hfAuthPath | quote }}
  - name: API_HF_JWT_PUBLIC_KEY_URL
    value: {{ .Values.api.hfJwtPublicKeyUrl | quote }}
  - name: API_HF_JWT_ALGORITHM
    value: {{ .Values.api.hfJwtAlgorithm | quote }}
  - name: API_HF_TIMEOUT_SECONDS
    value: {{ .Values.api.hfTimeoutSeconds | quote }}
  - name: API_HF_WEBHOOK_SECRET
    {{- if .Values.secrets.hfWebhookSecret.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfWebhookSecret.secretName | quote }}
        key: WEBHOOK_SECRET
        optional: false
    {{- else }}
    value: {{ .Values.secrets.hfWebhookSecret.value }}
    {{- end }}
  - name: API_MAX_AGE_LONG
    value: {{ .Values.api.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.api.maxAgeShort | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.api.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.api.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.api.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.api.uvicornPort | quote }}
  volumeMounts:
  {{ include "volumeMountCachedAssetsRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRO" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.api.uvicornPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.api.uvicornPort }}
  ports:
  - containerPort: {{ .Values.api.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.api.resources | nindent 4 }}
{{- end -}}
