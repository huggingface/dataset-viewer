# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envCloudfront" -}}
- name: CLOUDFRONT_EXPIRATION_SECONDS
  value: {{ .Values.cloudfront.expirationSeconds | quote }}
{{- if not (and .Values.secrets.infisical.csi.enabled .Values.secrets.cloudfront.keyPairId.fromSecret) }}
- name: CLOUDFRONT_KEY_PAIR_ID
  {{- if .Values.secrets.cloudfront.keyPairId.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cloudfront.keyPairId.secretName | default (include "datasetsServer.infisical.secretName" $) | quote }}
      key: CLOUDFRONT_KEY_PAIR_ID
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cloudfront.keyPairId.value | quote }}
  {{- end }}
{{- end }}
{{- if not (and .Values.secrets.infisical.csi.enabled .Values.secrets.cloudfront.privateKey.fromSecret) }}
- name: CLOUDFRONT_PRIVATE_KEY
  {{- if .Values.secrets.cloudfront.privateKey.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cloudfront.privateKey.secretName | default (include "datasetsServer.infisical.secretName" $) | quote }}
      key: CLOUDFRONT_PRIVATE_KEY
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cloudfront.privateKey.value | quote }}
  {{- end }}
{{- end }}
{{- end -}}
