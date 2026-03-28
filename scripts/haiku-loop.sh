#!/usr/bin/env bash
# haiku-loop.sh — Start/stop the 5-minute haiku observation loop.
#
# Usage:
#   ./scripts/haiku-loop.sh start    # Start the loop in the background
#   ./scripts/haiku-loop.sh stop     # Stop the loop gracefully
#   ./scripts/haiku-loop.sh status   # Check if the loop is running
#   ./scripts/haiku-loop.sh once     # Run a single cycle (foreground)
#
# The loop runs haiku-cycle.sh every 5 minutes. It writes its PID to
# haiku-loop.pid for clean shutdown.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/haiku-loop.pid"
LOG_FILE="$PROJECT_DIR/haiku-loop-daemon.log"
INTERVAL=300  # 5 minutes in seconds

# Load API key from file if not set
if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -f "$HOME/.openrouter.key" ]]; then
  OPENROUTER_API_KEY="$(cat "$HOME/.openrouter.key" | tr -d '[:space:]')"
  export OPENROUTER_API_KEY
fi

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    else
      # Stale PID file
      rm -f "$PID_FILE"
      return 1
    fi
  fi
  return 1
}

do_start() {
  if is_running; then
    echo "Haiku loop is already running (PID $(cat "$PID_FILE"))"
    exit 0
  fi

  echo "Starting haiku loop (interval: ${INTERVAL}s)..."

  # Run the loop in a subshell, backgrounded and detached
  (
    SLEEP_PID=""
    cleanup() { rm -f "$PID_FILE"; [[ -n "$SLEEP_PID" ]] && kill "$SLEEP_PID" 2>/dev/null; exit 0; }
    trap cleanup SIGTERM SIGINT
    echo $BASHPID > "$PID_FILE"

    while true; do
      echo "[$(date -u --iso-8601=seconds)] Running haiku cycle..." >> "$LOG_FILE"
      if "$SCRIPT_DIR/haiku-cycle.sh" >> "$LOG_FILE" 2>&1; then
        echo "[$(date -u --iso-8601=seconds)] Cycle completed successfully" >> "$LOG_FILE"
      else
        echo "[$(date -u --iso-8601=seconds)] Cycle had errors (exit $?)" >> "$LOG_FILE"
      fi
      sleep "$INTERVAL" &
      SLEEP_PID=$!
      wait "$SLEEP_PID" 2>/dev/null || true
      SLEEP_PID=""
    done
  ) &
  disown

  # Wait briefly for PID file to be written
  sleep 1
  if is_running; then
    echo "Haiku loop started (PID $(cat "$PID_FILE"))"
    echo "Daemon log: $LOG_FILE"
    echo "Haiku log: $PROJECT_DIR/haiku-log.txt"
  else
    echo "ERROR: Failed to start haiku loop" >&2
    exit 1
  fi
}

do_stop() {
  if ! is_running; then
    echo "Haiku loop is not running"
    exit 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping haiku loop (PID $pid)..."
  kill "$pid"

  # Wait for process to exit
  local i=0
  while kill -0 "$pid" 2>/dev/null && [[ $i -lt 15 ]]; do
    sleep 1
    i=$((i + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "WARNING: Process didn't stop gracefully, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_FILE"
  echo "Haiku loop stopped"
}

do_status() {
  if is_running; then
    echo "Haiku loop is running (PID $(cat "$PID_FILE"))"
  else
    echo "Haiku loop is not running"
  fi
}

do_once() {
  exec "$SCRIPT_DIR/haiku-cycle.sh"
}

case "${1:-}" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  once)   do_once ;;
  *)
    echo "Usage: $0 {start|stop|status|once}"
    echo "  start   — Start the 5-minute haiku loop in the background"
    echo "  stop    — Stop the loop gracefully"
    echo "  status  — Check if the loop is running"
    echo "  once    — Run a single observation cycle (foreground)"
    exit 1
    ;;
esac
