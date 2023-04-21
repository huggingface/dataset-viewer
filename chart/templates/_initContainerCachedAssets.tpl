# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "initContainerCachedAssets" -}}
- name: prepare-cached-assets
  image: ubuntu:focal
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  command: ["/bin/sh", "-c"]
  args:
  - chown {{ .Values.uid }}:{{ .Values.gid }} /mounted-path;
  volumeMounts:
  - mountPath: /mounted-path
    mountPropagation: None
    name: data
    subPath: "{{ include "cachedAssets.subpath" . }}"
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
{{- end -}}
