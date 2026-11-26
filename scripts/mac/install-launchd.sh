#!/usr/bin/env bash
# Start the planner automatically when you log in (and restart it if it stops).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="com.renalplanner.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/RenalPlanner/logs"
mkdir -p "$(dirname "$PLIST")" "$LOGS"
cat >"$PLIST" <<PLISTXML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/mac/run.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StandardOutPath</key><string>$LOGS/planner.out.log</string>
  <key>StandardErrorPath</key><string>$LOGS/planner.err.log</string>
</dict>
</plist>
PLISTXML
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed. The planner runs at http://127.0.0.1:${RENAL_PORT:-8501} and starts at login."
echo "Remove with: launchctl bootout gui/\$(id -u)/$LABEL && rm $PLIST"
