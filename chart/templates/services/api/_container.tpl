# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerApi" -}}
- name: "{{ include "name" . }}-api"
  image: {{ .Values.dockerImage.services.api }}
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
  - name: COMMON_ASSETS_BASE_URL
    value: "{{ include "assets.baseUrl" . }}"
  - name: COMMON_HF_ENDPOINT
    value: {{ .Values.common.hfEndpoint | quote }}
  - name: COMMON_HF_TOKEN
    value: {{ .Values.secrets.hfToken | quote }}
  - name: COMMON_LOG_LEVEL
    value: {{ .Values.common.logLevel | quote }}
  - name: API_HF_AUTH_PATH
    value: {{ .Values.api.hfAuthPath | quote }}
  - name: API_MAX_AGE_LONG
    value: {{ .Values.api.maxAgeLong | quote }}
  - name: API_MAX_AGE_SHORT
    value: {{ .Values.api.maxAgeShort | quote }}
  - name: API_UVICORN_HOSTNAME
    value: {{ .Values.api.uvicornHostname | quote }}
  - name: API_UVICORN_NUM_WORKERS
    value: {{ .Values.api.uvicornNumWorkers | quote }}
  - name: API_UVICORN_PORT
    value: {{ .Values.api.uvicornPort | quote }}
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.api.prometheusMultiprocDirectory | quote }}
  volumeMounts:
  - mountPath: {{ .Values.cache.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: true
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
