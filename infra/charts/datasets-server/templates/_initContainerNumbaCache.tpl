{{- define "initContainerNumbaCache" -}}
- name: prepare-numba-cache
  image: ubuntu:focal
  imagePullPolicy: IfNotPresent
  command: ["/bin/sh", "-c"]
  args:
  - chown {{ .Values.uid }}:{{ .Values.gid }} /mounted-path;
  volumeMounts:
  - mountPath: /mounted-path
    mountPropagation: None
    name: nfs
    subPath: "{{ include "numba.cache.subpath" . }}"
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
{{- end -}}
