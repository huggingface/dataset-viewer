# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "dnsConfig" -}}
dnsConfig:
  options:
    - name: ndots
      value: "1"
{{- end -}}
