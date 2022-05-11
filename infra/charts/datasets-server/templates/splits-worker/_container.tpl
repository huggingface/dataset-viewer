{{- define "containerSplitsWorker" -}}
- name: "{{ include "name" . }}-splits-worker"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.storage.assetsDirectory | quote }}
  - name: DATASETS_BLOCKLIST
    value: {{ .Values.splitsWorker.datasetsBlocklist | quote }}
  - name: DATASETS_REVISION
    value: {{ .Values.splitsWorker.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.storage.cacheDirectory }}/datasets"
  - name: HF_MODULES_CACHE
    value: "{{ .Values.storage.cacheDirectory }}/modules"
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
    value: {{ include "mongodb.url" . }}
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
  image: "{{ .Values.splitsWorker.image.repository }}/{{ .Values.splitsWorker.image.name }}:{{ .Values.splitsWorker.image.tag }}"
  imagePullPolicy: {{ .Values.splitsWorker.image.pullPolicy }}
  volumeMounts:
  - mountPath: {{ .Values.storage.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.storage.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  # TODO: provide readiness and liveness probes
  # readinessProbe:
  #   tcpSocket:
  #     port: {{ .Values.splitsWorker.readinessPort }}
  # livenessProbe:
  #   tcpSocket:
  #     port: {{ .Values.splitsWorker.readinessPort }}
  resources:
    {{ toYaml .Values.splitsWorker.resources | nindent 4 }}
{{- end -}}
