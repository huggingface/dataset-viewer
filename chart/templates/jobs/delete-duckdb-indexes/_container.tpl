# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

{{- define "containerDeleteDuckdbIndexes" -}}
- name: "{{ include "name" . }}-delete-duckdb-indexes"
  image: {{ include "jobs.cacheMaintenance.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  securityContext:
    allowPrivilegeEscalation: false
  resources: {{ toYaml .Values.deleteDuckdbIndexes.resources | nindent 4 }}
  env:
    {{ include "envCache" . | nindent 2 }}
    {{ include "envCommon" . | nindent 2 }}
  - name: CACHE_MAINTENANCE_ACTION
    value: {{ .Values.deleteDuckdbIndexes.action | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.deleteDuckdbIndexes.log.level | quote }}
  - name: COMMITTER_HF_TOKEN
    {{- if .Values.secrets.appParquetConverterHfToken.fromSecret }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.appParquetConverterHfToken.secretName | quote }}
        key: PARQUET_CONVERTER_HF_TOKEN
        optional: false
    {{- else }}
    value: {{ .Values.secrets.appParquetConverterHfToken.value }}
    {{- end }}
{{- end -}}
