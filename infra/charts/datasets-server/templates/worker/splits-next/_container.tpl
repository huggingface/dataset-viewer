{{- define "containerWorkerSplitsNext" -}}
- name: "{{ include "name" . }}-worker-splits-next"
  env:
  - name: ASSETS_DIRECTORY
    value: {{ .Values.worker.splitsNext.assetsDirectory | quote }}
  - name: DATASETS_REVISION
    value: {{ .Values.worker.splitsNext.datasetsRevision | quote }}
  - name: HF_DATASETS_CACHE
    value: "{{ .Values.worker.splitsNext.cacheDirectory }}/datasets"
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
    value: {{ .Values.worker.splitsNext.logLevel | quote }}
  - name: MAX_JOB_RETRIES
    value: {{ .Values.worker.splitsNext.maxJobRetries | quote }}
  - name: MAX_JOBS_PER_DATASET
    value: {{ .Values.worker.splitsNext.maxJobsPerDataset | quote }}
  - name: MAX_LOAD_PCT
    value: {{ .Values.worker.splitsNext.maxLoadPct | quote }}
  - name: MAX_MEMORY_PCT
    value: {{ .Values.worker.splitsNext.maxMemoryPct | quote }}
  - name: MAX_SIZE_FALLBACK
    value: {{ .Values.worker.splitsNext.maxSizeFallback | quote }}
  - name: MIN_CELL_BYTES
    value: {{ .Values.worker.splitsNext.minCellBytes | quote }}
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
    value: {{ .Values.worker.splitsNext.numbaCacheDirectory | quote }}
  - name: ROWS_MAX_BYTES
    value: {{ .Values.worker.splitsNext.rowsMaxBytes | quote }}
  - name: ROWS_MAX_NUMBER
    value: {{ .Values.worker.splitsNext.rowsMaxNumber | quote }}
  - name: ROWS_MIN_NUMBER
    value: {{ .Values.worker.splitsNext.rowsMinNumber| quote }}
  - name: WORKER_SLEEP_SECONDS
    value: {{ .Values.worker.splitsNext.workerleepSeconds | quote }}
  - name: WORKER_QUEUE
    # Job queue the worker will pull jobs from: 'datasets' or 'splits'
    value: "splits_responses"
  image: {{ .Values.dockerImage.worker.splitsNext }}
  imagePullPolicy: IfNotPresent
  volumeMounts:
  - mountPath: {{ .Values.worker.splitsNext.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.splitsNext.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.datasets.subpath" . }}"
    readOnly: false
  - mountPath: {{ .Values.worker.splitsNext.numbaCacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.numba.subpath" . }}"
    readOnly: false
  securityContext:
    allowPrivilegeEscalation: false
  # TODO: provide readiness and liveness probes
  # readinessProbe:
  #   tcpSocket:
  #     port: {{ .Values.worker.splitsNext.readinessPort }}
  # livenessProbe:
  #   tcpSocket:
  #     port: {{ .Values.worker.splitsNext.readinessPort }}
  resources:
    {{ toYaml .Values.worker.splitsNext.resources | nindent 4 }}
{{- end -}}
