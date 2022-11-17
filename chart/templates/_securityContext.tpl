# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "securityContext" -}}
runAsUser: {{ .Values.uid }}
runAsGroup: {{ .Values.gid }}
runAsNonRoot: true
{{- end -}}
