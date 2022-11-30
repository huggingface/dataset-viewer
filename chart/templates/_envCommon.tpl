# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCommon" -}}
- name: COMMON_ASSETS_BASE_URL
  value: "{{ include "assets.baseUrl" . }}"
- name: COMMON_HF_ENDPOINT
  value: {{ .Values.common.hfEndpoint | quote }}
- name: HF_ENDPOINT # see https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411
  value: {{ .Values.common.hfEndpoint | quote }}
- name: COMMON_LOG_LEVEL
  value: {{ .Values.common.logLevel | quote }}
{{- end -}}

{{- define "envCommonToken" -}}
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
{{- end -}}

{{- define "envCommonTokenFrancky" -}}
- name: COMMON_HF_TOKEN
  {{- if .Values.secrets.tokenFrancky.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.tokenFrancky.secretName | quote }}
      key: HF_TOKEN
      optional: false
  {{- else }}
  value: {{ .Values.secrets.tokenFrancky.value }}
  {{- end }}
{{- end -}}
