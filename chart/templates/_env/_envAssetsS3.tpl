# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envAssetsS3" -}}
- name: ASSETS_S3_BUCKET
  value: {{ .Values.assetsS3.bucket | quote }}
- name: ASSETS_S3_REGION
  value: {{ .Values.assetsS3.region | quote }}
- name: ASSETS_S3_FOLDER_NAME
  value: {{ .Values.assetsS3.folderName | quote }}
- name: ASSETS_S3_ACCESS_KEY_ID
  {{- if .Values.secrets.s3.accessKeyId.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.s3.accessKeyId.secretName | quote }}
      key: AWS_ACCESS_KEY_ID
      optional: false
  {{- else }}
  value: {{ .Values.secrets.s3.accessKeyId.value | quote }}
  {{- end }}
- name: ASSETS_S3_SECRET_ACCESS_KEY
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
