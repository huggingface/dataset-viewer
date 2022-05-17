{{- define "containerAdmin" -}}
- name: "{{ include "name" . }}-admin"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.splitsWorker.assetsDirectory | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.splitsWorker.logLevel | quote }}
  - name: MONGO_CACHE_DATABASE
    value: {{ .Values.mongodb.cacheDatabase | quote }}
  - name: MONGO_QUEUE_DATABASE
    value: {{ .Values.mongodb.queueDatabase | quote }}
  - name: MONGO_URL
  {{- if .Values.mongodb.enabled }}
    value: mongodb://{{.Release.Name}}-mongodb
  {{- else }}
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.mongoUrl | quote }}
        key: MONGO_URL
        optional: false
  {{- end }}
  image: "{{ .Values.admin.image.repository }}/{{ .Values.admin.image.name }}:{{ .Values.docker.tag }}"
  imagePullPolicy: {{ .Values.admin.image.pullPolicy }}
  volumeMounts:
  - mountPath: {{ .Values.admin.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  # TODO: provide readiness and liveness probes
  # readinessProbe:
  #   tcpSocket:
  #     port: {{ .Values.admin.readinessPort }}
  # livenessProbe:
  #   tcpSocket:
  #     port: {{ .Values.admin.readinessPort }}
  resources:
    {{ toYaml .Values.admin.resources | nindent 4 }}
{{- end -}}
