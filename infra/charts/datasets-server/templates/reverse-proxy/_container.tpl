{{- define "containerReverseProxy" -}}
- name: "{{ include "name" . }}-reverse-proxy"    
  image: {{ .Values.dockerImage.reverseProxy }}
  imagePullPolicy: IfNotPresent
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.reverseProxy.assetsDirectory | quote }}
  - name: CACHE_DIRECTORY
    value: {{ .Values.reverseProxy.cacheDirectory | quote }}
  - name: CACHE_INACTIVE
    value: {{ .Values.reverseProxy.cacheInactive | quote }}
  - name: CACHE_MAX_SIZE
    value: {{ .Values.reverseProxy.cacheMaxSize | quote }}
  - name: CACHE_ZONE_SIZE
    value: {{ .Values.reverseProxy.cacheZoneSize | quote }}
  - name: HOST
    value: {{ .Values.reverseProxy.host | quote }}
  - name: PORT
    value: {{ .Values.reverseProxy.port | quote }}
  - name: TARGET_URL
    value: {{ include "api.url" . | quote }}
  volumeMounts:
  - name: nginx-templates
    mountPath: /etc/nginx/templates
    mountPropagation: None
    readOnly: true
  - mountPath: {{ .Values.reverseProxy.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: true
  - mountPath: {{ .Values.reverseProxy.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "nginx.cache.subpath" . }}"
    readOnly: false
  readinessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.readinessPort }}
  livenessProbe:
    tcpSocket:
      port: {{ .Values.reverseProxy.readinessPort }}
  ports:
  - containerPort: {{ .Values.reverseProxy.port }}
    name: http
    protocol: TCP
  resources:
    {{ toYaml .Values.reverseProxy.resources | nindent 4 }}
{{- end -}}
