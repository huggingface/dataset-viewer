# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerMongodbMigration" -}}
- name: "{{ include "name" . }}-mongodb-migration"
  image: {{ .Values.dockerImage.jobs.mongodbMigration }}
  imagePullPolicy: IfNotPresent
  env:
  {{ include "envCache" . | nindent 2 }}
  {{ include "envQueue" . | nindent 2 }}
  {{ include "envCommon" . | nindent 2 }}
  {{ include "envCommonToken" . | nindent 2 }}
  - name: MONGODB_MIGRATION_MONGO_DATABASE
    value: {{ .Values.mongodbMigration.mongoDatabase | quote }}
  - name: MONGODB_MIGRATION_MONGO_URL
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
  volumeMounts:
  {{ include "volumeMountAssetsRO" . | nindent 2 }}
  securityContext:
    allowPrivilegeEscalation: false  
  resources: {{ toYaml .Values.mongodbMigration.resources | nindent 4 }}
{{- end -}}
