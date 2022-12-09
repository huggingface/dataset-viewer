# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerStorageAdmin" -}}
- name: "{{ include "name" . }}-storage-admin"
  image: ubuntu:focal
  imagePullPolicy: IfNotPresent
  volumeMounts: 
  - mountPath: /data
    mountPropagation: None
    name: data
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
  resources: {{ toYaml .Values.storageAdmin.resources | nindent 4 }}
  command:
  - 'sleep'
  - 'infinity'
{{- end -}}
