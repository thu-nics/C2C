#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-recipe/eval_recipe/workflow_eval.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate c2c

while IFS= read -r line; do
  run_name="${line%%|||*}"
  cmd="${line#*|||}"
  if [[ -z "${cmd}" || "${cmd}" == "${line}" ]]; then
    echo "Skipping invalid command line: ${line}" >&2
    continue
  fi
  echo "Running ${run_name}: ${cmd}"
  eval "${cmd}"
done < <(
  python - "${CONFIG_PATH}" <<'PY'
import shlex
import sys
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("Missing dependency: PyYAML is required to read the config.\n")
    sys.exit(1)

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
if not config:
    sys.stderr.write(f"Empty config: {config_path}\n")
    sys.exit(1)

base = config.get("base", {}) or {}
runs = config.get("runs", []) or []

if not runs:
    sys.stderr.write(f"No runs defined in: {config_path}\n")
    sys.exit(1)

flag_keys = {"resume", "judge_only", "enable_thinking", "stream", "patch"}

for run in runs:
    if not isinstance(run, dict):
        continue
    name = run.get("name", "run")
    args = deepcopy(base)
    for key, value in run.items():
        if key == "name":
            continue
        args[key] = value

    cmd = ["python", "script/workflow/evaluation.py"]
    for key, value in args.items():
        if value is None:
            continue
        arg = "--" + key.replace("_", "-")
        normalized_key = key.replace("-", "_")
        if normalized_key in flag_keys:
            if value:
                cmd.append(arg)
            continue
        if isinstance(value, bool):
            cmd.extend([arg, str(value).lower()])
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            cmd.append(arg)
            cmd.extend(str(v) for v in value)
            continue
        cmd.extend([arg, str(value)])

    print(f"{name}|||{shlex.join(cmd)}")
PY
)
