# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "containerRows" -}}
- name: "{{ include "name" . }}-rows"
  image: {{ include "services.rows.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCachedAssets" . | nindent 2 }}
  {{ include "envCache" . | nindent 2 }}
  {{ include "envParquetMetadata" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envNumba" . | nindent 2 }}
  # service
  - name: API_HF_AUTH_PATH
    value: {{ .Values.rows.hfAuthPath | quote }}
  - name: API_HF_JWT_PUBLIC_KEY_URL
    value: {{ .Values.rows.hfJwtPublicKeyUrl | quote }}
  - name: API_HF_JWT_ALGORITHM
    value: {{ .Values.rows.hfJwtAlgorithm | quote }}
  - name: API_HF_TIMEOUT_SECONDS
    value: {{ .Values.rows.hfTimeoutSeconds | quote }}
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
    value: {{ .Values.rows.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.rows.maxAgeShort | quote }}
  # prometheus
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.rows.prometheusMultiprocDirectory | quote }}
  # uvicorn
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.rows.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.rows.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.rows.uvicornPort | quote }}
  volumeMounts:
  {{ include "volumeMountCachedAssetsRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRO" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.rows.uvicornPort }}
  livenessProbe:
    failureThreshold: 30
    periodSeconds: 5
    httpGet:
      path: /healthcheck
      port: {{ .Values.rows.uvicornPort }}
  ports:
  - containerPort: {{ .Values.rows.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.rows.resources | nindent 4 }}
{{- end -}}
