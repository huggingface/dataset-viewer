# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCachedAssets" -}}
- name: CACHED_ASSETS_BASE_URL
  value: "{{ include "cachedAssets.baseUrl" . }}"
- name: CACHED_ASSETS_FOLDER_NAME
  value: {{ .Values.cachedAssets.folderName | quote }}
- name: CACHED_ASSETS_STORAGE_ROOT
  value: {{ .Values.cachedAssets.storageRoot | quote }}
- name: CACHED_ASSETS_STORAGE_PROTOCOL
  value: {{ .Values.cachedAssets.storageProtocol | quote }}
{{- end -}}
