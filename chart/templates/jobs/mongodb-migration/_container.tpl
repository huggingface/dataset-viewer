# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMongodbMigration" -}}
- name: "{{ include "name" . }}-mongodb-migration"
  image: {{ .Values.dockerImage.jobs.mongodbMigration }}
  imagePullPolicy: IfNotPresent
  env:
  - name: CACHE_ASSETS_DIRECTORY
    value: {{ .Values.cache.assetsDirectory | quote }}
  - name: CACHE_MONGO_DATABASE
    value: {{ .Values.cache.mongoDatabase | quote }}
  - name: CACHE_MONGO_URL
  {{- if .Values.secrets.mongoUrl.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl.secretName | quote }}
        key: MONGO_URL
        optional: false
  {{- else }}
    {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
    {{- else }}
    value: {{ .Values.secrets.mongoUrl.value }}
    {{- end }}
  {{- end }}
  - name: QUEUE_MONGO_DATABASE
    value: {{ .Values.queue.mongoDatabase | quote }}
  - name: QUEUE_MONGO_URL
  {{- if .Values.secrets.mongoUrl.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl.secretName | quote }}
        key: MONGO_URL
        optional: false
  {{- else }}
    {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
    {{- else }}
    value: {{ .Values.secrets.mongoUrl.value }}
    {{- end }}
  {{- end }}
  - name: COMMON_ASSETS_BASE_URL
    value: "{{ include "assets.baseUrl" . }}"
  - name: COMMON_HF_ENDPOINT
    value: {{ .Values.common.hfEndpoint | quote }}
  - name: COMMON_HF_TOKEN
  {{- if .Values.secrets.token.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.token.secretName | quote }}
        key: HF_TOKEN
        optional: false
  {{- else }}
    value: {{ .Values.secrets.token.value }}
  {{- end }}
  - name: COMMON_LOG_LEVEL
    value: {{ .Values.common.logLevel | quote }}
  - name: MONGODB_MIGRATION_MONGO_DATABASE
    value: {{ .Values.mongodbMigration.mongoDatabase | quote }}
  - name: MONGODB_MIGRATION_MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  volumeMounts:
  - mountPath: {{ .Values.cache.assetsDirectory | quote }}
    mountPropagation: None
    name: data
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false  
  resources:
    {{ toYaml .Values.mongodbMigration.resources | nindent 4 }}
{{- end -}}
