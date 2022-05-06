{{- define "splitsWorkerPodSpec" -}}
spec:
  containers:
  - name: hub-datasets-server-worker
    env:
    - name: ASSETS_DIRECTORY
      value: {{ .Values.storage.assetsDirectory | quote }}
    - name: DATASETS_BLOCKLIST
      value: {{ .Values.splitsWorker.datasetsBlocklist | quote }}
    - name: DATASETS_REVISION
      value: {{ .Values.splitsWorker.datasetsRevision | quote }}
    - name: HF_TOKEN
      # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
      # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
      valueFrom:
        secretKeyRef:
          name: datasets-server-secrets
          key: hfToken
          optional: false
    - name: LOG_LEVEL
      value: {{ .Values.splitsWorker.logLevel | quote }}
    - name: MAX_JOBS_PER_DATASET
      value: {{ .Values.splitsWorker.maxJobsPerDataset | quote }}
    - name: MAX_LOAD_PCT
      value: {{ .Values.splitsWorker.maxLoadPct | quote }}
    - name: MAX_MEMORY_PCT
      value: {{ .Values.splitsWorker.maxMemoryPct | quote }}
    - name: MAX_SIZE_FALLBACK
      value: {{ .Values.splitsWorker.maxSizeFallback | quote }}
    - name: MIN_CELL_BYTES
      value: {{ .Values.splitsWorker.minCellBytes | quote }}
    - name: MONGO_CACHE_DATABASE
      value: {{ .Values.mongodb.cacheDatabase | quote }}
    - name: MONGO_QUEUE_DATABASE
      value: {{ .Values.mongodb.queueDatabase | quote }}
    {{- if .Values.mongodb.enabled }}
    - name: MONGO_URL
      value: mongodb://{{.Release.Name}}-mongodb
    {{- end }}
    - name: ROWS_MAX_BYTES
      value: {{ .Values.splitsWorker.rowsMaxBytes | quote }}
    - name: ROWS_MAX_NUMBER
      value: {{ .Values.splitsWorker.rowsMaxNumber | quote }}
    - name: ROWS_MIN_NUMBER
      value: {{ .Values.splitsWorker.rowsMinNumber| quote }}
    - name: WORKER_SLEEP_SECONDS
      value: {{ .Values.splitsWorker.workerSleepSeconds | quote }}
    - name: WORKER_QUEUE
      # Job queue the worker will pull jobs from: 'datasets' or 'splits'
      value: "splits"
    image: "{{ .Values.api.image.repository }}/{{ .Values.api.image.name }}:{{ .Values.api.image.tag }}"
    imagePullPolicy: {{ .Values.api.image.pullPolicy }}
    volumeMounts:
    - mountPath: {{ .Values.storage.assetsDirectory | quote }}
      mountPropagation: None
      name: assets
      # in a subdirectory named as the chart (datasets-server/), and below it,
      # in a subdirectory named as the Release, so that Releases will not share the same assets/ dir
      subPath: "{{ include "name" . }}/{{ .Release.Name }}"
      # the workers have to write the assets to the disk
      readOnly: false
    # TODO: provide readiness and liveness probes
    # readinessProbe:
    #   tcpSocket:
    #     port: {{ .Values.api.readinessPort }}
    # livenessProbe:
    #   tcpSocket:
    #     port: {{ .Values.api.readinessPort }}
{{- end -}}
