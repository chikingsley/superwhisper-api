#!/bin/zsh
set -u

RUN_ID="canonical_20260522T112000Z"
JOB_DIR="/Volumes/simons-enjoyment/persian-asr/scribe-jobs/scribe-canonical-all-20260516T192536Z"
APP="/Applications/superwhisper.app"
APP_PROCESS="/Applications/superwhisper.app/Contents/MacOS/superwhisper"
LOG="/tmp/superwhisper-$RUN_ID-app-watchdog.log"
RSS_LIMIT_KB=$((200 * 1024))

: > /dev/null

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  echo "$(timestamp) $*" >> "$LOG"
}

controller_count=$(/usr/bin/pgrep -f "super-whisperer-workers --job-dir $JOB_DIR --run-id $RUN_ID" | /usr/bin/wc -l | /usr/bin/tr -d ' ')

if [[ "$controller_count" == "0" ]]; then
  log "controller not running; app watchdog idle"
  exit 0
fi

restart_app() {
  local name="$1"
  local user="$2"
  local uid="$3"
  local home="$4"
  local reason="$5"
  local pids="$6"

  log "$name restart: $reason"
  if [[ -n "$pids" ]]; then
    echo "$pids" | while read -r pid; do
      [[ -n "$pid" ]] || continue
      /bin/kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    echo "$pids" | while read -r pid; do
      [[ -n "$pid" ]] || continue
      /bin/kill -KILL "$pid" 2>/dev/null || true
    done
  fi

  /usr/bin/sudo /bin/launchctl asuser "$uid" \
    /usr/bin/sudo -u "$user" env \
    "HOME=$home" "USER=$user" "LOGNAME=$user" \
    open -g -a "$APP"
}

check_app() {
  local name="$1"
  local user="$2"
  local uid="$3"
  local home="$4"
  local rows
  local pids
  local max_rss

  rows=$(
    /bin/ps -axo pid=,user=,rss=,command= \
      | /usr/bin/awk -v user="$user" -v app_process="$APP_PROCESS" \
        '$2 == user && $4 == app_process {print $1, $3}'
  )

  if [[ -z "$rows" ]]; then
    restart_app "$name" "$user" "$uid" "$home" "app missing" ""
    return
  fi

  pids=$(echo "$rows" | /usr/bin/awk '{print $1}')
  max_rss=$(echo "$rows" | /usr/bin/awk 'max < $2 {max = $2} END {print max + 0}')

  if (( max_rss > RSS_LIMIT_KB )); then
    restart_app "$name" "$user" "$uid" "$home" "rss ${max_rss}KB > ${RSS_LIMIT_KB}KB" "$pids"
  fi
}

check_app "main" "simonpeacocks" "501" "/Users/simonpeacocks"
check_app "scribe1" "scribe1" "502" "/Users/scribe1"
check_app "scribe2" "scribe2" "503" "/Users/scribe2"
check_app "scribe3" "scribe3" "504" "/Users/scribe3"
check_app "scribe4" "scribe4" "505" "/Users/scribe4"

log "check complete"
exit 0
