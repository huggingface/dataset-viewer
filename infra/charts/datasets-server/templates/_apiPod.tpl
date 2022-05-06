{{- define "apiPodSpec" -}}
spec:
  containers:
  - name: hub-datasets-server-api
    env:
    - name: APP_HOSTNAME
      value: {{ .Values.api.appHostname | quote }}
    - name: APP_PORT
      value: {{ .Values.api.appPort | quote }}
    - name: ASSETS_DIRECTORY
      value: {{ .Values.storage.assetsDirectory | quote }}
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
    - mountPath: {{ .Values.storage.assetsDirectory | quote }}
      mountPropagation: None
      name: assets
      # in a subdirectory named as the chart (datasets-server/), and below it,
      # in a subdirectory named as the Release, so that Releases will not share the same assets/ dir
      subPath: "{{ include "name" . }}/{{ .Release.Name }}"
      # the api only requires read access to the assets
      readOnly: true
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
