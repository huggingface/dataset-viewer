{{- define "containerWorkerDatasets" -}}
- name: "{{ include "name" . }}-worker-datasets"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.worker.datasets.assetsDirectory | quote }}
  - name: DATASETS_REVISION
    value: {{ .Values.worker.datasets.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.worker.datasets.cacheDirectory }}/datasets"
  - name: HF_MODULES_CACHE
    value: "/tmp/modules-cache"
  # the size should remain so small that we don't need to worry about putting it on an external storage
  # see https://github.com/huggingface/datasets-server/issues/248
  - name: HF_TOKEN
    # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
    # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfToken | quote }}
        key: HF_TOKEN
        optional: false
  - name: LOG_LEVEL
    value: {{ .Values.worker.datasets.logLevel | quote }}
  - name: MAX_JOB_RETRIES
    value: {{ .Values.worker.datasets.maxJobRetries | quote }}
  - name: MAX_JOBS_PER_DATASET
    value: {{ .Values.worker.datasets.maxJobsPerDataset | quote }}
  - name: MAX_LOAD_PCT
    value: {{ .Values.worker.datasets.maxLoadPct | quote }}
  - name: MAX_MEMORY_PCT
    value: {{ .Values.worker.datasets.maxMemoryPct | quote }}
  - name: MAX_SIZE_FALLBACK
    value: {{ .Values.worker.datasets.maxSizeFallback | quote }}
  - name: MIN_CELL_BYTES
    value: {{ .Values.worker.datasets.minCellBytes | quote }}
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
  - name: NUMBA_CACHE_DIR
    value: {{ .Values.worker.datasets.numbaCacheDirectory | quote }}
  - name: ROWS_MAX_BYTES
    value: {{ .Values.worker.datasets.rowsMaxBytes | quote }}
  - name: ROWS_MAX_NUMBER
    value: {{ .Values.worker.datasets.rowsMaxNumber | quote }}
  - name: ROWS_MIN_NUMBER
    value: {{ .Values.worker.datasets.rowsMinNumber| quote }}
  - name: WORKER_SLEEP_SECONDS
    value: {{ .Values.worker.datasets.workerSleepSeconds | quote }}
  - name: WORKER_QUEUE
    # Job queue the worker will pull jobs from: 'datasets' or 'splits'
    value: "datasets"
  image: {{ .Values.dockerImage.worker.datasets }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.worker.datasets.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.datasets.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.datasets.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  # TODO: provide readiness and liveness probes
  # readinessProbe:
  #   tcpSocket:
  #     port: {{ .Values.worker.datasets.readinessPort }}
  # livenessProbe:
  #   tcpSocket:
  #     port: {{ .Values.worker.datasets.readinessPort }}
  resources:
    {{ toYaml .Values.worker.datasets.resources | nindent 4 }}
{{- end -}}
