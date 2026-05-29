#!/bin/zsh
set -u

RUN_ID="canonical_20260522T112000Z"

/bin/launchctl bootout system /Library/LaunchDaemons/com.simon.superwhisperer.app-watchdog.plist 2>/dev/null || true

pids=$(
  {
    /usr/bin/pgrep -f '/Applications/superwhisper.app/Contents/MacOS/superwhisper' || true
    /usr/bin/pgrep -f "superwhisper-api-workers .*--run-id $RUN_ID" || true
    /usr/bin/pgrep -f "/Users/simonpeacocks/GitHub/superwhisper-api/.venv/bin/python3 .*superwhisper-api --paths-file .*$RUN_ID" || true
    /usr/bin/pgrep -f "sudo .*superwhisper-api --paths-file .*$RUN_ID" || true
  } | /usr/bin/sort -nu
)

if [[ -n "$pids" ]]; then
  echo "$pids" | while read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" == "$$" ]] && continue
    /bin/kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 5

  survivors=$(
    {
      /usr/bin/pgrep -f '/Applications/superwhisper.app/Contents/MacOS/superwhisper' || true
      /usr/bin/pgrep -f "superwhisper-api-workers .*--run-id $RUN_ID" || true
      /usr/bin/pgrep -f "/Users/simonpeacocks/GitHub/superwhisper-api/.venv/bin/python3 .*superwhisper-api --paths-file .*$RUN_ID" || true
      /usr/bin/pgrep -f "sudo .*superwhisper-api --paths-file .*$RUN_ID" || true
    } | /usr/bin/sort -nu
  )

  if [[ -n "$survivors" ]]; then
    echo "$survivors" | while read -r pid; do
      [[ -n "$pid" ]] || continue
      [[ "$pid" == "$$" ]] && continue
      /bin/kill -KILL "$pid" 2>/dev/null || true
    done
  fi
fi

/bin/ps -axo pid,user,rss,command | /usr/bin/grep -E 'superwhisper-api-workers|/Applications/superwhisper.app/Contents/MacOS/superwhisper' | /usr/bin/grep -v grep || true
