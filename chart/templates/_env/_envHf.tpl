# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envHf" -}}
- name: API_HF_AUTH_PATH
  value: {{ .Values.hf.authPath | quote }}
- name: API_HF_JWT_PUBLIC_KEY_URL
  value: {{ .Values.hf.jwtPublicKeyUrl | quote }}
- name: API_HF_JWT_ADDITIONAL_PUBLIC_KEYS
  {{- if .Values.secrets.hfJwtAdditionalPublicKeys.fromSecret }}
  valueFrom:
    secretKeyRef:
    name: {{ .Values.secrets.hfJwtAdditionalPublicKeys.secretName | quote }}
    key: API_HF_JWT_ADDITIONAL_PUBLIC_KEYS
    optional: false
  {{- else }}
  value: {{ .Values.secrets.hfJwtAdditionalPublicKeys.value | quote }}
  {{- end }}
- name: API_HF_JWT_ALGORITHM
  value: {{ .Values.hf.jwtAlgorithm | quote }}
- name: API_HF_TIMEOUT_SECONDS
  value: {{ .Values.hf.timeoutSeconds | quote }}
- name: API_HF_WEBHOOK_SECRET
  {{- if .Values.secrets.hfWebhookSecret.fromSecret }}
  valueFrom:
    secretKeyRef:
    name: {{ .Values.secrets.hfWebhookSecret.secretName | quote }}
    key: WEBHOOK_SECRET
    optional: false
  {{- else }}
  value: {{ .Values.secrets.hfWebhookSecret.value | quote }}
  {{- end }}
{{- end -}}
