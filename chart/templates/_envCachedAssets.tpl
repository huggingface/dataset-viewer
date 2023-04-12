# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envCachedAssets" -}}
- name: CACHED_ASSETS_BASE_URL
  value: "{{ include "cachedAssets.baseUrl" . }}"
- name: CACHED_ASSETS_STORAGE_DIRECTORY
  value: {{ .Values.cachedAssets.storageDirectory . | quote }}
- name: CACHED_ASSETS_CLEAN_CACHE_PROBA
  value: "{{ include "cachedAssets.cleanCacheProba" . | quote }}"
- name: CACHED_ASSETS_KEEP_FIRST_ROWS_NUMBER
  value: "{{ include "cachedAssets.keepFirstRowsNumber" . | quote }}"
- name: CACHED_ASSETS_KEEP_MOST_RECENT_ROWS_NUMBER
  value: "{{ include "cachedAssets.keepMostRecentRowsNumber" . | quote }}"
- name: CACHED_ASSETS_MAX_CLEANED_ROWS_NUMBER
  value: "{{ include "cachedAssets.maxCleanedRowsNumber" . | quote }}"
{{- end -}}
