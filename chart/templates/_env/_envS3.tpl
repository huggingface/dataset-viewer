# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envS3" -}}
- name: S3_BUCKET
  value: {{ .Values.s3.bucket | quote }}
- name: S3_REGION
  value: {{ .Values.s3.region | quote }}
- name: S3_ACCESS_KEY_ID
  {{- if .Values.secrets.s3.accessKeyId.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.s3.accessKeyId.secretName | quote }}
      key: AWS_ACCESS_KEY_ID
      optional: false
  {{- else }}
  value: {{ .Values.secrets.s3.accessKeyId.value | quote }}
  {{- end }}
- name: S3_SECRET_ACCESS_KEY
  {{- if .Values.secrets.s3.secretAccessKey.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.s3.secretAccessKey.secretName | quote }}
      key: AWS_SECRET_ACCESS_KEY
      optional: false
  {{- else }}
  value: {{ .Values.secrets.s3.secretAccessKey.value | quote }}
  {{- end }}
{{- end -}}
