{{- define "initContainerCache" -}}
- name: prepare-cache
  image: ubuntu:focal
  imagePullPolicy: IfNotPresent
  command: ["/bin/sh", "-c"]
  args:
  - chown {{ .Values.uid }}:{{ .Values.gid }} {{ .Values.storage.cacheDirectory | quote }};
  volumeMounts:
  - mountPath: {{ .Values.storage.cacheDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "cache.subpath" . }}"
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
{{- end -}}
