{{- define "apiPodSpec" -}}
spec:
  containers:
  - name: hub-datasets-server-api
    env:
      - name: APP_HOSTNAME
        value: {{ .Values.api.appHostname | quote }}
      - name: APP_PORT
        value: {{ .Values.api.appPort | quote }}
      {{- if .Values.global.storage.enabled }}
      - name: ASSETS_DIRECTORY
        value: {{ .Values.global.storage.assetsDirectory | quote }}
      {{- end }}
      - name: LOG_LEVEL
        value: {{ .Values.api.logLevel | quote }}
      - name: MAX_AGE_LONG_SECONDS
        value: {{ .Values.api.maxAgeLongSeconds | quote }}
      - name: MAX_AGE_SHORT_SECONDS
        value: {{ .Values.api.maxAgeShortSeconds | quote }}
      {{- if .Values.mongodb.enabled }}
      - name: MONGO_CACHE_DATABASE
        value: {{ .Values.mongodb.cacheDatabase | quote }}
      - name: MONGO_QUEUE_DATABASE
        value: {{ .Values.mongodb.queueDatabase | quote }}
      - name: MONGO_URL
        value: mongodb://{{.Release.Name}}-mongodb
      {{- end }}
      - name: WEB_CONCURRENCY
        value: {{ .Values.api.webConcurrency | quote }}
    image: "{{ .Values.image.repository }}/{{ .Values.image.name }}:{{ .Values.image.tag }}"
    imagePullPolicy: {{ .Values.image.pullPolicy }}
    {{- if .Values.global.storage.enabled }}
    volumeMounts:
    - mountPath: {{ .Values.global.storage.assetsDirectory | quote }}
      name: data
    {{- end }}
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
{{- end -}}
