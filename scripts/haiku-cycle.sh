#!/usr/bin/env bash
# haiku-cycle.sh — Run one observation cycle: launch all three observers in
# parallel, wait for completion, aggregate results, and commit to git.
#
# Usage: ./scripts/haiku-cycle.sh
#
# Environment:
#   OPENROUTER_API_KEY  — required by observers (or set in ~/.openrouter.key)
#   HAIKU_TIMESTAMP     — override the cycle timestamp (default: generated)
#   SKIP_GIT            — if set, skip git commit/push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HAIKU_DIR="$PROJECT_DIR/haikus"
LOG_FILE="$PROJECT_DIR/haiku-log.txt"

# Load API key from file if not set
if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -f "$HOME/.openrouter.key" ]]; then
  OPENROUTER_API_KEY="$(cat "$HOME/.openrouter.key" | tr -d '[:space:]')"
  export OPENROUTER_API_KEY
fi

# Shared timestamp so all three observers write consistently named files
export HAIKU_TIMESTAMP="${HAIKU_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}"
TS="$HAIKU_TIMESTAMP"

# Ensure directories exist
mkdir -p "$HAIKU_DIR"

echo "[haiku-cycle] Starting cycle at $TS"

# --- Launch all three observers in parallel ---
pids=()
observers=(disk process network)

for obs in "${observers[@]}"; do
  "$SCRIPT_DIR/${obs}-observer.sh" &
  pids+=($!)
done

# --- Wait for all to complete, track failures ---
failures=()
for i in "${!observers[@]}"; do
  if ! wait "${pids[$i]}"; then
    failures+=("${observers[$i]}")
    echo "[haiku-cycle] WARNING: ${observers[$i]}-observer failed" >&2
  fi
done

if [[ ${#failures[@]} -eq ${#observers[@]} ]]; then
  echo "[haiku-cycle] ERROR: All observers failed" >&2
  exit 1
fi

# --- Aggregate haikus into the log ---
HUMAN_TS="$(echo "$TS" | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)-\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3T\4:\5:\6/')"

{
  echo "=== ${HUMAN_TS} ==="
  for obs in "${observers[@]}"; do
    haiku_file="$HAIKU_DIR/${obs}-${TS}.haiku"
    if [[ -f "$haiku_file" ]]; then
      # Extract just the haiku lines (after the second --- frontmatter delimiter)
      haiku_text="$(awk '/^---$/{c++;next} c==2{if(/^$/){next} if(/^---$|^observations:|^<!--|^  /){exit} print}' "$haiku_file")"
      echo "[${obs}]"
      echo "$haiku_text"
    else
      echo "[${obs}]"
      echo "(observer failed)"
    fi
    echo ""
  done
  echo "==="
  echo ""
} >> "$LOG_FILE"

# --- Commit and push to git (if remote configured and not skipped) ---
if [[ -z "${SKIP_GIT:-}" ]]; then
  cd "$PROJECT_DIR"
  if git rev-parse --git-dir > /dev/null 2>&1; then
    # Check if there's a remote configured
    if git remote get-url origin > /dev/null 2>&1; then
      echo "[haiku-cycle] Committing to git..."
      git add haikus/ haiku-log.txt 2>/dev/null || true
      if git diff --cached --quiet; then
        echo "[haiku-cycle] No changes to commit"
      else
        git commit -m "haiku: batch ${TS}" 2>/dev/null || true
        echo "[haiku-cycle] Pushing to origin..."
        git push origin main 2>&1 || echo "[haiku-cycle] WARNING: Push failed (check remote configuration)"
      fi
    else
      echo "[haiku-cycle] No remote configured, skipping git push"
    fi
  fi
fi

echo "[haiku-cycle] Cycle complete. Haikus appended to $LOG_FILE"
if [[ ${#failures[@]} -gt 0 ]]; then
  echo "[haiku-cycle] Partial failures: ${failures[*]}"
fi
