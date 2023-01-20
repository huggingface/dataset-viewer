# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMongodbMigration" -}}
- name: "{{ include "name" . }}-mongodb-migration"
  image: {{ include "jobs.mongodbMigration.image" . }}
  imagePullPolicy: IfNotPresent
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  - name: MONGODB_MIGRATION_MONGO_DATABASE
    value: {{ .Values.mongodbMigration.mongoDatabase | quote }}
  - name: MONGODB_MIGRATION_MONGO_URL
    {{ include "datasetServer.mongo.url" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.mongodbMigration.resources | nindent 4 }}
{{- end -}}
