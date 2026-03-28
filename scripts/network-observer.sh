#!/usr/bin/env bash
# network-observer.sh — Gathers network state, asks an LLM to write a haiku about it,
# and writes the result atomically to the haikus/ directory.
#
# Requires: OPENROUTER_API_KEY environment variable
# Optional: OPENROUTER_MODEL (default: deepseek/deepseek-v3.2)
#           HAIKU_OUTPUT_DIR (default: /home/erik/autohaiku/haikus)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${HAIKU_OUTPUT_DIR:-$PROJECT_DIR/haikus}"
MODEL="${OPENROUTER_MODEL:-deepseek/deepseek-v3.2}"
TIMESTAMP="${HAIKU_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Gather network observations ──────────────────────────────────────────────

gather_network_data() {
  local obs=""

  obs+="=== IP Addresses ===
"
  obs+="$(ip addr show 2>/dev/null | grep 'inet ' || echo 'ip addr unavailable')
"

  obs+="
=== Ping 8.8.8.8 ===
"
  obs+="$(ping -c 1 -W 2 8.8.8.8 2>&1 || echo 'ping to 8.8.8.8 failed')
"

  obs+="
=== Ping google.com ===
"
  obs+="$(ping -c 1 -W 2 google.com 2>&1 || echo 'ping to google.com failed')
"

  obs+="
=== Active Connections (ss) ===
"
  obs+="$(ss -tunp 2>/dev/null | head -10 || echo 'ss unavailable')
"

  obs+="
=== Default Route ===
"
  obs+="$(ip route show default 2>/dev/null || echo 'no default route')
"

  obs+="
=== Interface Traffic (/proc/net/dev) ===
"
  obs+="$(cat /proc/net/dev 2>/dev/null | tail -n +3 || echo '/proc/net/dev unavailable')
"

  obs+="
=== DNS Resolution ===
"
  obs+="$(host google.com 2>&1 | head -3 || dig google.com +short 2>&1 | head -3 || echo 'DNS resolution unavailable')
"

  printf '%s' "$obs"
}

echo "Gathering network data..." >&2
NETWORK_DATA="$(gather_network_data)"

# ── Call LLM to compose haiku ────────────────────────────────────────────────

SYSTEM_PROMPT='You are a haiku poet who writes about computer network observations. You will receive raw network diagnostic data and must compose a single haiku (5-7-5 syllable structure) that references something specific from the data — an IP address, a latency measurement, a connection count, an interface name, or similar concrete detail. Do NOT write a generic haiku. The haiku must be grounded in what was actually observed.

Respond with ONLY three lines of the haiku, nothing else. No titles, no explanations, no quotes.'

USER_PROMPT="Here is the current network state of this machine. Write a haiku about it.

$NETWORK_DATA"

# Build JSON payload safely using jq
PAYLOAD="$(jq -n \
  --arg model "$MODEL" \
  --arg system "$SYSTEM_PROMPT" \
  --arg user "$USER_PROMPT" \
  '{
    model: $model,
    messages: [
      { role: "system", content: $system },
      { role: "user", content: $user }
    ],
    temperature: 0.9,
    max_tokens: 100
  }'
)"

echo "Calling $MODEL via OpenRouter..." >&2
RESPONSE="$(curl -s -f --max-time 30 \
  https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://github.com/autohaiku" \
  -H "X-Title: autohaiku-network-observer" \
  -d "$PAYLOAD"
)"

# Extract the haiku text from the response
HAIKU="$(echo "$RESPONSE" | jq -r '.choices[0].message.content' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

if [[ -z "$HAIKU" || "$HAIKU" == "null" ]]; then
  echo "ERROR: Failed to get haiku from LLM. Response:" >&2
  echo "$RESPONSE" >&2
  exit 1
fi

# ── Write haiku file atomically ──────────────────────────────────────────────

FINAL="$OUTPUT_DIR/network-$TIMESTAMP.haiku"
TMPFILE="$OUTPUT_DIR/.network-$TIMESTAMP.haiku.tmp"

cat > "$TMPFILE" <<EOF
---
observer: network
timestamp: $TIMESTAMP
---
$HAIKU
EOF

# Append raw observation data as a metadata comment
{
  echo ""
  echo "<!-- observation data"
  echo "$NETWORK_DATA"
  echo "-->"
} >> "$TMPFILE"

# Atomic rename
mv "$TMPFILE" "$FINAL"

echo "Haiku written to $FINAL" >&2
echo "---"
cat "$FINAL"
