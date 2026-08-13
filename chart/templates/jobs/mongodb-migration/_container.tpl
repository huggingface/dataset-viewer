# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMongodbMigration" -}}
- name: "{{ include "name" . }}-mongodb-migration"
  image: {{ include "jobs.mongodbMigration.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envLog" . | nindent 2 }}
  {{ include "envSecrets" . | nindent 2 }}
  - name: DATABASE_MIGRATIONS_MONGO_DATABASE
    value: {{ .Values.mongodbMigration.mongoDatabase | quote }}
  {{- if not (and .Values.secrets.infisical.csi.enabled .Values.secrets.mongoUrl.fromSecret) }}
  - name: DATABASE_MIGRATIONS_MONGO_URL
    {{ include "datasetServer.mongo.url" . | nindent 4 }}
  {{- end }}
  {{- include "datasetsServer.csi.volumeMountBlock" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.mongodbMigration.resources | nindent 4 }}
{{- end -}}
