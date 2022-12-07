# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerReverseProxy" -}}
- name: "{{ include "name" . }}-reverse-proxy"
  image: {{ .Values.dockerImage.reverseProxy }}
  imagePullPolicy: IfNotPresent
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.assets.storageDirectory | quote }}
  - name: HOST
    value: {{ .Values.reverseProxy.host | quote }}
  - name: PORT
    value: {{ .Values.reverseProxy.port | quote }}
  - name: URL_ADMIN
    value: {{ include "admin.url" . | quote }}
  - name: URL_API
    value: {{ include "api.url" . | quote }}
  volumeMounts:
  {{ include "volumeMountAssetsRO" . | nindent 2 }}
  - name: nginx-templates
    mountPath: /etc/nginx/templates
    mountPropagation: None
    readOnly: true
  - name: error-pages
    mountPath: /error-pages
    mountPropagation: None
    readOnly: true
  - name: static-files
    mountPath: /static-files
    mountPropagation: None
    readOnly: true
  readinessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.readinessPort }}
  ports:
  - containerPort: {{ .Values.reverseProxy.port }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.reverseProxy.resources | nindent 4 }}
{{- end -}}
