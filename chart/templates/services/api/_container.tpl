# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerApi" -}}
- name: "{{ include "name" . }}-api"
  image: {{ .Values.dockerImage.services.api }}
  imagePullPolicy: IfNotPresent
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envCommonToken" . | nindent 2 }}
  # service
  - name: API_HF_AUTH_PATH
    value: {{ .Values.api.hfAuthPath | quote }}
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
  volumeMounts: {{ include "volumeMountAssetsRO" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.api.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.api.readinessPort }}
  ports:
  - containerPort: {{ .Values.api.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.api.resources | nindent 4 }}
{{- end -}}
