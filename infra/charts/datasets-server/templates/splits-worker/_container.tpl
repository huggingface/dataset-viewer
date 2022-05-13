{{- define "containerSplitsWorker" -}}
- name: "{{ include "name" . }}-splits-worker"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.splitsWorker.assetsDirectory | quote }}
  - name: DATASETS_BLOCKLIST
    value: {{ .Values.splitsWorker.datasetsBlocklist | quote }}
  - name: DATASETS_REVISION
    value: {{ .Values.splitsWorker.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.splitsWorker.cacheDirectory }}/datasets"
  - name: HF_MODULES_CACHE
    value: "{{ .Values.splitsWorker.cacheDirectory }}/modules"
  - name: HF_TOKEN
    # see https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret
    # and https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables
    valueFrom:
      secretKeyRef:
        name: {{ .Values.secrets.hfToken | quote }}
        key: HF_TOKEN
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
  image: "{{ .Values.splitsWorker.image.repository }}/{{ .Values.splitsWorker.image.name }}:{{ .Values.docker.tag }}"
  imagePullPolicy: {{ .Values.splitsWorker.image.pullPolicy }}
  volumeMounts:
  - mountPath: {{ .Values.splitsWorker.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.splitsWorker.cacheDirectory | quote }}
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
