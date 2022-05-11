{{- define "containerApi" -}}
- name: "{{ include "name" . }}-api"
  env:
  - name: APP_HOSTNAME
    value: {{ .Values.api.appHostname | quote }}
  - name: APP_PORT
    value: {{ .Values.api.appPort | quote }}
  - name: ASSETS_DIRECTORY
    value: {{ .Values.api.assetsDirectory | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.api.logLevel | quote }}
  - name: MAX_AGE_LONG_SECONDS
    value: {{ .Values.api.maxAgeLongSeconds | quote }}
  - name: MAX_AGE_SHORT_SECONDS
    value: {{ .Values.api.maxAgeShortSeconds | quote }}
  - name: MONGO_CACHE_DATABASE
    value: {{ .Values.mongodb.cacheDatabase | quote }}
  - name: MONGO_QUEUE_DATABASE
    value: {{ .Values.mongodb.queueDatabase | quote }}
  {{- if .Values.mongodb.enabled }}
  - name: MONGO_URL
    value: mongodb://{{.Release.Name}}-mongodb
  {{- end }}
  - name: WEB_CONCURRENCY
    value: {{ .Values.api.webConcurrency | quote }}
  image: "{{ .Values.api.image.repository }}/{{ .Values.api.image.name }}:{{ .Values.api.image.tag }}"
  imagePullPolicy: {{ .Values.api.image.pullPolicy }}
  volumeMounts:
  - mountPath: {{ .Values.api.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: true
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.api.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.api.readinessPort }}
  ports:
  - containerPort: {{ .Values.api.appPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.api.resources | nindent 4 }}
{{- end -}}
