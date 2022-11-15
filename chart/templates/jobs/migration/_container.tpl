# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMigration" -}}
- name: "{{ include "name" . }}-migration"
  image: {{ .Values.dockerImage.jobs.migration }}
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
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfToken | quote }}
        key: HF_TOKEN
        optional: false
  - name: COMMON_LOG_LEVEL
    value: {{ .Values.common.logLevel | quote }}
  - name: MIGRATION_MONGO_DATABASE
    value: {{ .Values.migration.mongoDatabase | quote }}
  - name: MIGRATION_MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources:
    {{ toYaml .Values.migration.resources | nindent 4 }}
{{- end -}}
