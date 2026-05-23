#!/bin/zsh
set -euo pipefail

REPO="/Users/simonpeacocks/GitHub/super-whisperer-cli"
JOB_DIR="/Volumes/simons-enjoyment/persian-asr/scribe-jobs/scribe-canonical-all-20260516T192536Z"
RUN_ID="resume_20260520T122211Z"
WORKERS_CLI="$REPO/.venv/bin/super-whisperer-workers"
LOG="$JOB_DIR/results/superwhisper-$RUN_ID-recycler.log"
CONTROLLER_LOG="$JOB_DIR/results/superwhisper-$RUN_ID-controller.log"
LOCK_DIR="/tmp/superwhisper-$RUN_ID-recycler.lock"

mkdir -p "$(dirname "$LOG")"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') recycler already running; exiting" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

{
  echo
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') recycle start ====="

  pids=$(
    {
      pgrep -f '/Applications/superwhisper.app/Contents/MacOS/superwhisper' || true
      pgrep -f "super-whisperer-workers --job-dir $JOB_DIR --run-id $RUN_ID" || true
      pgrep -f "/Users/simonpeacocks/GitHub/super-whisperer-cli/.venv/bin/python3 .*super-whisperer --paths-file .*$RUN_ID" || true
      pgrep -f "sudo .*super-whisperer --paths-file .*$RUN_ID" || true
    } | sort -nu
  )

  if [[ -n "$pids" ]]; then
    echo "TERM:"
    echo "$pids"
    echo "$pids" | while read -r pid; do
      [[ -n "$pid" ]] || continue
      [[ "$pid" == "$$" ]] && continue
      /bin/kill -TERM "$pid" 2>/dev/null || true
    done

    sleep 5

    survivors=$(
      {
        pgrep -f '/Applications/superwhisper.app/Contents/MacOS/superwhisper' || true
        pgrep -f "super-whisperer-workers --job-dir $JOB_DIR --run-id $RUN_ID" || true
        pgrep -f "/Users/simonpeacocks/GitHub/super-whisperer-cli/.venv/bin/python3 .*super-whisperer --paths-file .*$RUN_ID" || true
        pgrep -f "sudo .*super-whisperer --paths-file .*$RUN_ID" || true
      } | sort -nu
    )

    if [[ -n "$survivors" ]]; then
      echo "KILL:"
      echo "$survivors"
      echo "$survivors" | while read -r pid; do
        [[ -n "$pid" ]] || continue
        [[ "$pid" == "$$" ]] && continue
        /bin/kill -KILL "$pid" 2>/dev/null || true
      done
      sleep 3
    fi
  else
    echo "no existing run processes found"
  fi

  echo "launching $RUN_ID"
  cd "$REPO"
  "$WORKERS_CLI" --job-dir "$JOB_DIR" --run-id "$RUN_ID" >> "$CONTROLLER_LOG" 2>&1 &
  controller_pid=$!
  echo "controller pid: $controller_pid"

  sleep 60
  echo "post-launch counts:"
  /usr/bin/wc -l "$JOB_DIR"/results/scribev2."$RUN_ID".*.jsonl 2>/dev/null || true
  echo "post-launch processes:"
  /bin/ps -axo user,pid,stat,etime,%cpu,%mem,rss,command | /opt/homebrew/bin/rg '/Applications/superwhisper\.app/Contents/MacOS/superwhisper|super-whisperer-workers --job-dir .*resume_20260520T122211Z|super-whisperer --paths-file .*resume_20260520T122211Z' || true
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') recycle end ====="
} >> "$LOG" 2>&1
