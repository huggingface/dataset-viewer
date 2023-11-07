# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerReverseProxy" -}}
- name: "{{ include "name" . }}-reverse-proxy"
  image: {{ include "reverseproxy.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  env:
  - name: OPENAPI_FILE
    value: {{ .Values.reverseProxy.openapiFile | quote }}
  - name: HOST
    value: {{ .Values.reverseProxy.host | quote }}
  - name: PORT
    value: {{ .Values.reverseProxy.port | quote }}
  - name: URL_ADMIN
    value: {{ include "admin.url" . | quote }}
  - name: URL_API
    value: {{ include "api.url" . | quote }}
  - name: URL_ROWS
    value: {{ include "rows.url" . | quote }}
  - name: URL_SEARCH
    value: {{ include "search.url" . | quote }}
  - name: URL_SSE_API
    value: {{ include "sseApi.url" . | quote }}
  volumeMounts:
  - name: nginx-templates
    mountPath: /etc/nginx/templates
    mountPropagation: None
    readOnly: true
  - name: error-pages
    mountPath: /error-pages
    mountPropagation: None
    readOnly: true
  readinessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.port }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.port }}
  ports:
  - containerPort: {{ .Values.reverseProxy.port }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.reverseProxy.resources | nindent 4 }}
{{- end -}}
