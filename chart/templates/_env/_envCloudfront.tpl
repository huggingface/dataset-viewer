# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envS3" -}}
- name: CLOUDFRONT_EXPIRATION_SECONDS
  value: {{ .Values.cloudfront.expirationSeconds | quote }}
- name: CLOUDFRONT_KEY_PAIR_ID
  {{- if .Values.secrets.cloudfront.keyPairId.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cloudfront.keyPairId.secretName | quote }}
      key: CLOUDFRONT_KEY_PAIR_ID
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cloudfront.keyPairId.value | quote }}
  {{- end }}
- name: CLOUDFRONT_PRIVATE_KEY
  {{- if .Values.secrets.cloudfront.privateKey.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cloudfront.privateKey.secretName | quote }}
      key: CLOUDFRONT_PRIVATE_KEY
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cloudfront.privateKey.value | quote }}
  {{- end }}
{{- end -}}
