# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerAdmin" -}}
- name: "{{ include "name" . }}-admin"
  image: {{ .Values.dockerImage.services.admin }}
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
  - name: ADMIN_HF_ORGANIZATION
    value: {{ .Values.admin.hfOrganization | quote }}
  - name: ADMIN_CACHE_REPORTS_NUM_RESULTS
    value: {{ .Values.admin.cacheReportsNumResults | quote }}
  - name: ADMIN_HF_WHOAMI_PATH
    value: {{ .Values.admin.hfWhoamiPath | quote }}
  - name: ADMIN_MAX_AGE
    value: {{ .Values.admin.maxAge | quote }}
  - name: ADMIN_UVICORN_HOSTNAME
    value: {{ .Values.admin.uvicornHostname | quote }}
  - name: ADMIN_UVICORN_NUM_WORKERS
    value: {{ .Values.admin.uvicornNumWorkers | quote }}
  - name: ADMIN_UVICORN_PORT
    value: {{ .Values.admin.uvicornPort | quote }}
  - name: PROMETHEUS_MULTIPROC_DIR
    value:  {{ .Values.admin.prometheusMultiprocDirectory | quote }}
  volumeMounts:
  - mountPath: {{ .Values.cache.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.admin.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.admin.readinessPort }}
  ports:
  - containerPort: {{ .Values.admin.uvicornPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.admin.resources | nindent 4 }}
{{- end -}}
