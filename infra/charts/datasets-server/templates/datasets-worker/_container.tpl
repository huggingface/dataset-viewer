{{- define "containerDatasetsWorker" -}}
- name: "{{ include "name" . }}-datasets-worker"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.datasetsWorker.assetsDirectory | quote }}
  - name: DATASETS_REVISION
    value: {{ .Values.datasetsWorker.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.datasetsWorker.cacheDirectory }}/datasets"
  - name: HF_MODULES_CACHE
    value: "{{ .Values.datasetsWorker.cacheDirectory }}/modules"
  - name: HF_TOKEN
    # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
    # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfToken | quote }}
        key: HF_TOKEN
        optional: false
  - name: LOG_LEVEL
    value: {{ .Values.datasetsWorker.logLevel | quote }}
  - name: MAX_JOB_RETRIES
    value: {{ .Values.datasetsWorker.maxJobRetries | quote }}
  - name: MAX_JOBS_PER_DATASET
    value: {{ .Values.datasetsWorker.maxJobsPerDataset | quote }}
  - name: MAX_LOAD_PCT
    value: {{ .Values.datasetsWorker.maxLoadPct | quote }}
  - name: MAX_MEMORY_PCT
    value: {{ .Values.datasetsWorker.maxMemoryPct | quote }}
  - name: MAX_SIZE_FALLBACK
    value: {{ .Values.datasetsWorker.maxSizeFallback | quote }}
  - name: MIN_CELL_BYTES
    value: {{ .Values.datasetsWorker.minCellBytes | quote }}
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
  - name: ROWS_MAX_BYTES
    value: {{ .Values.datasetsWorker.rowsMaxBytes | quote }}
  - name: ROWS_MAX_NUMBER
    value: {{ .Values.datasetsWorker.rowsMaxNumber | quote }}
  - name: ROWS_MIN_NUMBER
    value: {{ .Values.datasetsWorker.rowsMinNumber| quote }}
  - name: WORKER_SLEEP_SECONDS
    value: {{ .Values.datasetsWorker.workerSleepSeconds | quote }}
  - name: WORKER_QUEUE
    # Job queue the worker will pull jobs from: 'datasets' or 'splits'
    value: "datasets"
  image: {{ .Values.dockerImage.datasetsWorker }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.datasetsWorker.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.datasetsWorker.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  # TODO: provide readiness and liveness probes
  # readinessProbe:
  #   tcpSocket:
  #     port: {{ .Values.datasetsWorker.readinessPort }}
  # livenessProbe:
  #   tcpSocket:
  #     port: {{ .Values.datasetsWorker.readinessPort }}
  resources:
    {{ toYaml .Values.datasetsWorker.resources | nindent 4 }}
{{- end -}}
