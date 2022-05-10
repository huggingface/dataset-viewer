{{- define "initContainerAssets" -}}
- name: prepare-assets
  image: alpine:latest
  imagePullPolicy: IfNotPresent
  command: ["/bin/sh", "-c"]
  args:
  - chown {{ .Values.uid }}:{{ .Values.gid }} {{ .Values.storage.assetsDirectory | quote }};
  volumeMounts:
  - mountPath: {{ .Values.storage.assetsDirectory | quote }}
    mountPropagation: None
    name: nfs
    subPath: "{{ include "assets.subpath" . }}"
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
{{- end -}}
