# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

{{/*
Where the application reads its secrets from. Only a path, never a secret.
*/}}
{{- define "envSecrets" -}}
{{- if .Values.secrets.infisical.csi.enabled }}
- name: SECRETS_DIR
  value: {{ .Values.secrets.infisical.csi.mountPath | quote }}
{{- end }}
{{- end -}}

{{/*
The read-only tmpfs volume a workload's secrets are mounted from. Call with a dict of the root context
and the workload name: {{ include "datasetsServer.csi.volume" (dict "context" $ "workload" "api") }}
*/}}
{{- define "datasetsServer.csi.volume" -}}
{{- if .context.Values.secrets.infisical.csi.enabled }}
- name: secrets
  csi:
    driver: secrets-store.csi.k8s.io
    readOnly: true
    volumeAttributes:
      secretProviderClass: {{ include "name" .context }}-{{ .workload }}-secrets
{{- end }}
{{- end -}}

{{- define "datasetsServer.csi.volumeMount" -}}
{{- if .Values.secrets.infisical.csi.enabled }}
- name: secrets
  mountPath: {{ .Values.secrets.infisical.csi.mountPath | quote }}
  readOnly: true
{{- end }}
{{- end -}}

{{/*
Whole-block variants, for pod specs and containers that have no volumes or volumeMounts of their own:
the key itself has to sit inside the condition, or an empty one is rendered when the mode is off.
*/}}
{{- define "datasetsServer.csi.volumeBlock" -}}
{{- if .context.Values.secrets.infisical.csi.enabled }}
volumes:
  {{- include "datasetsServer.csi.volume" . | nindent 2 }}
{{- end }}
{{- end -}}

{{- define "datasetsServer.csi.volumeMountBlock" -}}
{{- if .Values.secrets.infisical.csi.enabled }}
volumeMounts:
  {{- include "datasetsServer.csi.volumeMount" . | nindent 2 }}
{{- end }}
{{- end -}}
