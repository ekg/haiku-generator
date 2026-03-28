#!/usr/bin/env bash
# process-observer.sh — Observe running processes and system load, then write a haiku.
#
# Usage: ./scripts/process-observer.sh
#
# Environment:
#   OPENROUTER_API_KEY  — OpenRouter API key (falls back to WG_API_KEY)
#   HAIKU_MODEL         — model to use (default: deepseek/deepseek-v3.2)
#   HAIKU_DIR           — output directory (default: /home/erik/autohaiku/haikus)

set -euo pipefail

HAIKU_DIR="${HAIKU_DIR:-/home/erik/autohaiku/haikus}"
API_KEY="${OPENROUTER_API_KEY:-${WG_API_KEY:-}}"
MODEL="${HAIKU_MODEL:-deepseek/deepseek-v3.2}"
TIMESTAMP="${HAIKU_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}"

if [[ -z "$API_KEY" ]]; then
  echo "Error: No API key found. Set OPENROUTER_API_KEY or WG_API_KEY." >&2
  exit 1
fi

mkdir -p "$HAIKU_DIR"

# --- Gather process/system observations ---

obs_uptime="$(uptime 2>/dev/null || echo 'uptime unavailable')"
obs_top_cpu="$(ps aux --sort=-%cpu 2>/dev/null | head -6 || echo 'ps cpu unavailable')"
obs_top_mem="$(ps aux --sort=-%mem 2>/dev/null | head -6 || echo 'ps mem unavailable')"
obs_free="$(free -h 2>/dev/null || echo 'free unavailable')"
obs_nproc="$(nproc 2>/dev/null || echo 'nproc unavailable')"

observations="UPTIME:
${obs_uptime}

TOP PROCESSES BY CPU:
${obs_top_cpu}

TOP PROCESSES BY MEMORY:
${obs_top_mem}

MEMORY USAGE:
${obs_free}

CPU COUNT: ${obs_nproc}"

# --- Build the LLM prompt ---

prompt="You are a haiku poet observing a computer's running processes and system load.

Here are the current observations:

${observations}

Write exactly ONE haiku (three lines: 5 syllables, 7 syllables, 5 syllables).
The haiku MUST reference something specific from the observations — a process name, load average, memory figure, or other concrete detail. Do not write a generic haiku.

Reply with ONLY the three lines of the haiku, nothing else. No title, no explanation, no quotes."

# --- Call the OpenRouter API ---

# Escape the prompt for JSON
json_prompt="$(printf '%s' "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')"

request_body="$(cat <<ENDJSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "user", "content": ${json_prompt}}
  ],
  "temperature": 0.9,
  "max_tokens": 100
}
ENDJSON
)"

response="$(curl -s -f -X POST "https://openrouter.ai/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "$request_body")"

# Extract the haiku text from the response
haiku_text="$(printf '%s' "$response" | python3 -c '
import sys, json
data = json.load(sys.stdin)
content = data["choices"][0]["message"]["content"].strip()
print(content)
')"

if [[ -z "$haiku_text" ]]; then
  echo "Error: LLM returned empty response." >&2
  exit 1
fi

# --- Write the haiku file atomically ---

FINAL="${HAIKU_DIR}/process-${TIMESTAMP}.haiku"
TMPFILE="${HAIKU_DIR}/.process-${TIMESTAMP}.haiku.tmp"

cat > "$TMPFILE" <<HAIKU
---
observer: process
timestamp: ${TIMESTAMP}
---
${haiku_text}
HAIKU

mv "$TMPFILE" "$FINAL"

echo "Wrote haiku to ${FINAL}"
echo "---"
cat "$FINAL"
