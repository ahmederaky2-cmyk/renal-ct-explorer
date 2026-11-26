#!/usr/bin/env bash
# Start the private planner (patient mode) on this Mac.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${RENAL_ENV_FILE:-$HOME/RenalPlanner/planner.env}"
set -a; source "$ENV_FILE"; set +a
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE="${STREAMLIT_SERVER_MAX_UPLOAD_SIZE:-2048}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
cd "$REPO"
# caffeinate keeps the Mac awake (on power) while the planner is running.
exec caffeinate -is .venv/bin/streamlit run app.py \
  --server.address "${RENAL_BIND_ADDRESS:-127.0.0.1}" \
  --server.port "${RENAL_PORT:-8501}" \
  --server.headless true
