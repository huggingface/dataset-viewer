{{- define "containerAdmin" -}}
- name: "{{ include "name" . }}-admin"
  env:
  - name: APP_HOSTNAME
    value: {{ .Values.admin.appHostname | quote }}
  - name: APP_NUM_WORKERS
    value: {{ .Values.admin.appNumWorkers | quote }}
  - name: APP_PORT
    value: {{ .Values.admin.appPort | quote }}
  - name: ASSETS_DIRECTORY
    value: {{ .Values.admin.assetsDirectory | quote }}
  - name: CACHE_REPORTS_NUM_RESULTS
    value: {{ .Values.admin.cacheReportsNumResults | quote }}
  - name: LOG_LEVEL
    value: {{ .Values.admin.logLevel | quote }}
  - name: MAX_AGE_SHORT_SECONDS
    value: {{ .Values.admin.maxAgeShortSeconds | quote }}
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
  image: {{ .Values.dockerImage.admin }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.admin.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.admin.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.admin.readinessPort }}
  ports:
  - containerPort: {{ .Values.admin.appPort }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.admin.resources | nindent 4 }}
{{- end -}}
