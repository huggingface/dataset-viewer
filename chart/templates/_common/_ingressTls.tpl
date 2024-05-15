{{/*
TLS part of ingress template
*/}}
{{- define "ingress.tls" -}}
{{- if include "hf.common.ingress.certManagerRequest" ( dict "annotations" .annotations ) }}
tls:
  - hosts:
      - {{ include "datasetsServer.ingress.hostname" . }}
    secretName: {{ printf "%s-tls" (include "datasetsServer.ingress.hostname" $) }}
{{- else if .Values.ingress.tls -}}
{{- with .Values.ingress.tls }}
tls:
  {{- tpl (toYaml .) $ | nindent 2 }}
{{- end }}
{{- end }}
{{- end }}
