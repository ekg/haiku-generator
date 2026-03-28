#!/usr/bin/env bash
set -euo pipefail

# Disk Observer Haiku Agent
# Gathers disk/filesystem state and writes a haiku about it via OpenRouter LLM.

HAIKU_DIR="/home/erik/autohaiku/haikus"
TIMESTAMP="${HAIKU_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}"
FINAL="$HAIKU_DIR/disk-${TIMESTAMP}.haiku"
TMPFILE="$HAIKU_DIR/.disk-${TIMESTAMP}.haiku.tmp"

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-${WG_API_KEY:-}}"
OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1/chat/completions}"
MODEL="${DISK_OBSERVER_MODEL:-deepseek/deepseek-v3.2}"

if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "ERROR: No API key. Set OPENROUTER_API_KEY or WG_API_KEY." >&2
  exit 1
fi

mkdir -p "$HAIKU_DIR"

# --- Gather disk observations ---
obs_df=$(df -h 2>/dev/null || true)
obs_du=$(du -sh "$HAIKU_DIR" 2>/dev/null || echo "haikus dir: not yet created")
obs_inodes=$(df -i / 2>/dev/null | tail -1 || true)
obs_recent=$(find /home/erik -maxdepth 2 -mmin -5 -type f 2>/dev/null | head -10 || true)
obs_recent_haikus=$(ls -lt "$HAIKU_DIR"/*.haiku 2>/dev/null | head -5 || echo "no haikus yet")

OBSERVATIONS="## Disk Usage (df -h)
$obs_df

## Haiku Directory Size
$obs_du

## Root Inode Usage
$obs_inodes

## Recently Modified Files (last 5 min)
${obs_recent:-none}

## Most Recent Haikus
$obs_recent_haikus"

# --- Call LLM to compose haiku ---
SYSTEM_PROMPT='You are a haiku poet observing a computer'\''s disk and filesystem state. You write haikus in strict 5-7-5 syllable format. Your haikus must reference something specific from the observation data — a percentage, a directory name, a file count, something concrete. Do not write generic haikus.

Respond with ONLY the three lines of the haiku, nothing else. No quotes, no explanation, no title.'

USER_PROMPT="Here is the current disk/filesystem state of this machine. Write a single haiku (5-7-5 syllables) that references something specific you see in this data.

$OBSERVATIONS"

# Build JSON payload safely with jq
PAYLOAD=$(jq -n \
  --arg model "$MODEL" \
  --arg system "$SYSTEM_PROMPT" \
  --arg user "$USER_PROMPT" \
  '{
    model: $model,
    messages: [
      { role: "system", content: $system },
      { role: "user", content: $user }
    ],
    max_tokens: 100,
    temperature: 0.9
  }')

RESPONSE=$(curl -s --max-time 30 \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://github.com/autohaiku" \
  -H "X-Title: autohaiku-disk-observer" \
  -d "$PAYLOAD" \
  "$OPENROUTER_URL")

# Extract haiku text from response
HAIKU_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty' 2>/dev/null)

if [[ -z "$HAIKU_TEXT" ]]; then
  ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error.message // .error // "unknown error"' 2>/dev/null)
  echo "ERROR: LLM returned no haiku. Response: $ERROR_MSG" >&2
  echo "Full response: $RESPONSE" >&2
  exit 1
fi

# Trim whitespace and ensure exactly 3 lines
HAIKU_TEXT=$(echo "$HAIKU_TEXT" | sed '/^$/d' | head -3)
LINE_COUNT=$(echo "$HAIKU_TEXT" | wc -l)
if [[ "$LINE_COUNT" -ne 3 ]]; then
  echo "WARNING: LLM returned $LINE_COUNT lines instead of 3" >&2
fi

# --- Write haiku file atomically ---
cat > "$TMPFILE" <<EOF
---
observer: disk
timestamp: $TIMESTAMP
---
$HAIKU_TEXT

---
observations: |
$(echo "$OBSERVATIONS" | sed 's/^/  /')
EOF

mv "$TMPFILE" "$FINAL"

echo "Haiku written to: $FINAL"
echo "---"
echo "$HAIKU_TEXT"
