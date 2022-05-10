{{/*
Expand the name of the chart.
*/}}
{{- define "name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "selectorLabels" -}}
app.kubernetes.io/name: {{ include "name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "labels" -}}
helm.sh/chart: {{ include "chart" . }}
{{ include "selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
release: {{ $.Release.Name | quote }}
heritage: {{ $.Release.Service | quote }}
chart: "{{ include "name" . }}"
{{- end }}

{{- define "labels.api" -}}
{{ include "labels" . }}
app: "{{ .Release.Name }}-api"
{{- end -}}

{{- define "labels.datasetsWorker" -}}
{{ include "labels" . }}
app: "{{ .Release.Name }}-datasets-worker"
{{- end -}}

{{- define "labels.splitsWorker" -}}
{{ include "labels" . }}
app: "{{ .Release.Name }}-splits-worker"
{{- end -}}

{{/*
The assets/ subpath in the NFS
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same assets/ dir
*/}}
{{- define "assets.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "assets" }}
{{- end }}

{{/*
The cache/ subpath in the NFS
- in a subdirectory named as the chart (datasets-server/), and below it,
- in a subdirectory named as the Release, so that Releases will not share the same assets/ dir
*/}}
{{- define "cache.subpath" -}}
{{- printf "%s/%s/%s/" .Chart.Name .Release.Name "cache" }}
{{- end }}
