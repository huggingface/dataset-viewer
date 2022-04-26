#!/bin/bash

# see https://github.com/microsoft/pylance-release/discussions/2712#discussioncomment-2603488

shopt -s dotglob
shopt -s nullglob

VENV_PATHS_ARR=(*/*/.venv/lib/*/site-packages/)
VENV_PATHS=$(printf ",\"%s\"" "${VENV_PATHS_ARR[@]}")
EXTRA_PATHS=${VENV_PATHS:1}

jq ".\"python.analysis.extraPaths\" = [${EXTRA_PATHS}]" .vscode/settings.json > .vscode/settings.tmp
mv .vscode/settings.tmp .vscode/settings.json 
