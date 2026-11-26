#!/usr/bin/env bash
# One-time setup of the private renal planner on an Apple Silicon Mac.
#   scripts/mac/setup.sh                 # environment, folders, TotalSegmentator-MLX weights
#   scripts/mac/setup.sh --with-atlasnet # also the 2.3 GB AbdomenAtlasNet tumour model
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HOME_DIR="${RENAL_HOME:-$HOME/RenalPlanner}"
WITH_ATLASNET=0
[[ "${1:-}" == "--with-atlasnet" ]] && WITH_ATLASNET=1

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Warning: not an Apple Silicon Mac; the MLX backend will be skipped (PyTorch is used)."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python manager) …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO"
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-patient.txt

mkdir -p "$HOME_DIR"/{cases,inbox,models,logs}
chmod 700 "$HOME_DIR" "$HOME_DIR/cases" "$HOME_DIR/inbox"

echo "Downloading TotalSegmentator-MLX weights …"
.venv/bin/python - <<PY || echo "MLX weights skipped (non-Apple-Silicon or offline)."
from huggingface_hub import snapshot_download
snapshot_download("MedMLX/totalsegmentator-mlx", local_dir="$HOME_DIR/models/totalsegmentator-mlx")
PY

if [[ $WITH_ATLASNET == 1 ]]; then
  echo "Downloading AbdomenAtlasNet-Tumors (CC BY-NC 4.0, ~2.3 GB) …"
  .venv/bin/python - <<PY
import zipfile
from huggingface_hub import hf_hub_download
path = hf_hub_download("AbdomenAtlas/AbdomenAtlasNet", "AbdomenAtlasNetTumors.zip")
zipfile.ZipFile(path).extractall("$HOME_DIR/models/AbdomenAtlasNetTumors")
PY
fi

ENV_FILE="$HOME_DIR/planner.env"
if [[ ! -f "$ENV_FILE" ]]; then
  PASSWORD="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(15))')"
  HASH="$(printf '%s' "$PASSWORD" | .venv/bin/python -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest())')"
  cat >"$ENV_FILE" <<ENV
# Renal planner settings (sourced by scripts/mac/run.sh). Keep this file private.
RENAL_PATIENT_MODE=1
RENAL_PATIENT_ROOT=$HOME_DIR/cases
RENAL_INBOX=$HOME_DIR/inbox
RENAL_RETENTION_DAYS=30
RENAL_APP_PASSWORD_SHA256=$HASH
RENAL_TOTALSEG_BACKEND=auto
RENAL_TOTALSEG_MLX_MODELS=$HOME_DIR/models/totalsegmentator-mlx
RENAL_ATLASNET_TUMOR_DIR=$HOME_DIR/models/AbdomenAtlasNetTumors
RENAL_DEVICE=auto
RENAL_CS_THRESHOLD_HU=400
# Optional validated KiPA22 ONNX vessel model (see training/README.md):
# RENAL_VESSEL_MODEL_DIR=$HOME_DIR/models/kipa22_vessel_unet_v1
# 127.0.0.1 = reachable only through the tunnel/locally; 0.0.0.0 = also on the LAN.
RENAL_BIND_ADDRESS=127.0.0.1
RENAL_PORT=8501
ENV
  chmod 600 "$ENV_FILE"
  echo
  echo "App password (store it in your password manager; shown once): $PASSWORD"
fi

echo
echo "Setup complete. Start with: scripts/mac/run.sh"
echo "Auto-start at login:        scripts/mac/install-launchd.sh"
