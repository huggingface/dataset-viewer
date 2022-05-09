{{- define "datasetsWorkerPodSpec" -}}
spec:
  containers:
  - name: "{{ include "instanceName" .}}-datasets-worker"
    env:
    # - name: ASSETS_DIRECTORY
    #   value: {{ .Values.storage.assetsDirectory | quote }}
    - name: DATASETS_BLOCKLIST
      value: {{ .Values.datasetsWorker.datasetsBlocklist | quote }}
    - name: DATASETS_REVISION
      value: {{ .Values.datasetsWorker.datasetsRevision | quote }}
    - name: HF_TOKEN
      # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
      # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
      valueFrom:
        secretKeyRef:
          name: datasets-server-secrets
          key: hfToken
          optional: false
    - name: LOG_LEVEL
      value: {{ .Values.datasetsWorker.logLevel | quote }}
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
    {{- if .Values.mongodb.enabled }}
    - name: MONGO_URL
      value: mongodb://{{.Release.Name}}-mongodb
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
    image: "{{ .Values.datasetsWorker.image.repository }}/{{ .Values.datasetsWorker.image.name }}:{{ .Values.datasetsWorker.image.tag }}"
    imagePullPolicy: {{ .Values.datasetsWorker.image.pullPolicy }}
    # volumeMounts:
    # - mountPath: {{ .Values.storage.assetsDirectory | quote }}
    #   mountPropagation: None
    #   name: assets
    #   # in a subdirectory named as the chart (datasets-server/), and below it,
    #   # in a subdirectory named as the Release, so that Releases will not share the same assets/ dir
    #   subPath: "{{ include "name" . }}/{{ .Release.Name }}"
    #   # the workers have to write the assets to the disk
    #   readOnly: false
    # TODO: provide readiness and liveness probes
    # readinessProbe:
    #   tcpSocket:
    #     port: {{ .Values.datasetsWorker.readinessPort }}
    # livenessProbe:
    #   tcpSocket:
    #     port: {{ .Values.datasetsWorker.readinessPort }}
{{- end -}}
