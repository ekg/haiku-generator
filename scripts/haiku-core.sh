#!/usr/bin/env bash
# haiku-core.sh — Reusable core pipeline functions for the haiku news system.
#
# Source this file from cycle tasks or scripts:
#   source scripts/haiku-core.sh
#
# Provides three pipeline functions:
#   fetch_news       — Retrieve a news item from an API source
#   generate_haiku   — Transform a news item into a haiku using an LLM
#   publish_haiku    — Save the haiku to the haiku stream (haikus/ dir + log)
#
# Environment variables (set before sourcing or exporting):
#   OPENROUTER_API_KEY  — Required for LLM calls (or WG_API_KEY)
#   HAIKU_MODEL         — LLM model (default: deepseek/deepseek-v3.2)
#   HAIKU_DIR           — Output directory (default: /home/erik/autohaiku/haikus)
#   HAIKU_LOG_FILE      — Aggregate log (default: /home/erik/autohaiku/haiku-log.txt)
#   NEWS_API_URL        — News API endpoint (default: https://hacker-news.firebaseio.com/v0)
#   OPENROUTER_URL      — OpenRouter endpoint (default: https://openrouter.ai/api/v1/chat/completions)

set -euo pipefail

# ── Configuration defaults ────────────────────────────────────────────────────

SCRIPT_DIR_CORE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR_CORE="$(dirname "$SCRIPT_DIR_CORE")"

HAIKU_DIR="${HAIKU_DIR:-$PROJECT_DIR_CORE/haikus}"
HAIKU_LOG_FILE="${HAIKU_LOG_FILE:-$PROJECT_DIR_CORE/haiku-log.txt}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-${WG_API_KEY:-}}"
OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1/chat/completions}"
HAIKU_MODEL="${HAIKU_MODEL:-deepseek/deepseek-v3.2}"
NEWS_API_URL="${NEWS_API_URL:-https://hacker-news.firebaseio.com/v0}"

# Load API key from file if not set
if [[ -z "$OPENROUTER_API_KEY" ]] && [[ -f "$HOME/.openrouter.key" ]]; then
  OPENROUTER_API_KEY="$(cat "$HOME/.openrouter.key" | tr -d '[:space:]')"
fi

# ── 1) fetch_news ─────────────────────────────────────────────────────────────
#
# Retrieves a news item from a news API source. By default uses the Hacker News
# Firebase API (free, no key required).
#
# Usage:
#   news_json=$(fetch_news)              # fetch a random top story
#   news_json=$(fetch_news <item_id>)    # fetch a specific story by ID
#
# Output: JSON object with keys: id, title, url, score, by, time
# Returns: 0 on success, 1 on failure (empty JSON {} on failure)
#
fetch_news() {
  local item_id="${1:-}"

  if [[ -z "$item_id" ]]; then
    # Pick a random item from the top stories list
    local top_stories
    top_stories=$(curl -sS --max-time 10 \
      "${NEWS_API_URL}/topstories.json" 2>/dev/null) || {
      echo '{"error": "Failed to fetch top stories"}' >&2
      echo '{}'
      return 1
    }

    # Pick one at random from the first 30
    local count
    count=$(echo "$top_stories" | jq 'length' 2>/dev/null) || count=0
    if [[ "$count" -eq 0 ]]; then
      echo '{"error": "No stories found"}' >&2
      echo '{}'
      return 1
    fi

    local max_idx=$(( count > 30 ? 30 : count ))
    local random_idx=$(( RANDOM % max_idx ))
    item_id=$(echo "$top_stories" | jq ".[$random_idx]" 2>/dev/null)
  fi

  # Fetch the specific story
  local item_json
  item_json=$(curl -sS --max-time 10 \
    "${NEWS_API_URL}/item/${item_id}.json" 2>/dev/null) || {
    echo "{\"error\": \"Failed to fetch item $item_id\"}" >&2
    echo '{}'
    return 1
  }

  if [[ -z "$item_json" || "$item_json" == "null" ]]; then
    echo '{"error": "Item not found"}' >&2
    echo '{}'
    return 1
  fi

  echo "$item_json"
}

# ── 2) generate_haiku ────────────────────────────────────────────────────────
#
# Transforms a news item into a haiku using an LLM via OpenRouter.
#
# Usage:
#   haiku_text=$(generate_haiku "$news_json")
#
# Input:  JSON string with at least a "title" field
# Output: Three lines of haiku text (5-7-5 syllables)
# Returns: 0 on success, 1 on failure
#
generate_haiku() {
  local news_json="$1"

  if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "ERROR: No API key. Set OPENROUTER_API_KEY or WG_API_KEY." >&2
    return 1
  fi

  # Extract news fields
  local title url score by
  title=$(echo "$news_json" | jq -r '.title // "untitled"' 2>/dev/null)
  url=$(echo "$news_json" | jq -r '.url // ""' 2>/dev/null)
  score=$(echo "$news_json" | jq -r '.score // ""' 2>/dev/null)
  by=$(echo "$news_json" | jq -r '.by // ""' 2>/dev/null)

  local news_summary="Title: ${title}"
  [[ -n "$url" && "$url" != "null" ]] && news_summary+="
URL: ${url}"
  [[ -n "$score" && "$score" != "null" ]] && news_summary+="
Score: ${score} points"
  [[ -n "$by" && "$by" != "null" ]] && news_summary+="
Posted by: ${by}"

  local system_prompt='You are a haiku poet who writes about current news and technology. Write a single haiku in strict 5-7-5 syllable format. The haiku must reference something specific from the news item — the subject, a name, a technology, or the sentiment. Be creative and evocative.

Respond with ONLY the three lines of the haiku. No quotes, no title, no explanation.'

  local user_prompt="Write a haiku about this news item:

${news_summary}"

  # Build JSON payload safely with jq
  local payload
  payload=$(jq -n \
    --arg model "$HAIKU_MODEL" \
    --arg system "$system_prompt" \
    --arg user "$user_prompt" \
    '{
      model: $model,
      messages: [
        { role: "system", content: $system },
        { role: "user", content: $user }
      ],
      max_tokens: 100,
      temperature: 0.9
    }')

  local response
  response=$(curl -sS --max-time 30 \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/autohaiku" \
    -H "X-Title: autohaiku-news" \
    -d "$payload" \
    "$OPENROUTER_URL") || {
    echo "ERROR: LLM API call failed" >&2
    return 1
  }

  # Extract haiku text from response
  local haiku_text
  haiku_text=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)

  if [[ -z "$haiku_text" || "$haiku_text" == "null" ]]; then
    local error_msg
    error_msg=$(echo "$response" | jq -r '.error.message // .error // "unknown error"' 2>/dev/null)
    echo "ERROR: LLM returned no haiku. Error: $error_msg" >&2
    return 1
  fi

  # Normalize: remove blank lines, take first 3 lines
  haiku_text=$(echo "$haiku_text" | sed '/^[[:space:]]*$/d' | head -3)
  local line_count
  line_count=$(echo "$haiku_text" | wc -l)
  if [[ "$line_count" -ne 3 ]]; then
    echo "WARNING: LLM returned $line_count lines instead of 3" >&2
  fi

  echo "$haiku_text"
}

# ── 3) publish_haiku ─────────────────────────────────────────────────────────
#
# Saves a haiku to the haiku stream: writes a .haiku file and appends to the log.
#
# Usage:
#   publish_haiku "$haiku_text" "$news_json" [timestamp]
#
# Args:
#   haiku_text  — The three lines of the haiku
#   news_json   — The original news JSON (for metadata)
#   timestamp   — Optional timestamp (default: current UTC)
#
# Side effects:
#   - Creates haikus/news-YYYYMMDD-HHMMSS.haiku (atomic write)
#   - Appends entry to haiku-log.txt
# Returns: 0 on success
#
publish_haiku() {
  local haiku_text="$1"
  local news_json="${2:-'{}'}"
  local timestamp="${3:-$(date -u +%Y%m%d-%H%M%S)}"

  if [[ -z "$haiku_text" ]]; then
    echo "ERROR: No haiku text to publish" >&2
    return 1
  fi

  mkdir -p "$HAIKU_DIR"

  # Extract news metadata for the haiku file frontmatter
  local title source_url
  title=$(echo "$news_json" | jq -r '.title // "unknown"' 2>/dev/null || echo "unknown")
  source_url=$(echo "$news_json" | jq -r '.url // ""' 2>/dev/null || echo "")

  local final_file="${HAIKU_DIR}/news-${timestamp}.haiku"
  local tmp_file="${HAIKU_DIR}/.news-${timestamp}.haiku.tmp"

  # Write to temp file first (atomic write pattern)
  cat > "$tmp_file" <<EOF
---
observer: news
timestamp: ${timestamp}
source_title: ${title}
source_url: ${source_url}
---
${haiku_text}
EOF

  # Atomic rename
  mv "$tmp_file" "$final_file"

  # Append to the aggregate log
  local human_ts
  human_ts=$(echo "$timestamp" | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)-\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3T\4:\5:\6/')
  {
    echo "=== ${human_ts} ==="
    echo "[news] ${title}"
    echo "${haiku_text}"
    echo ""
    echo "==="
    echo ""
  } >> "$HAIKU_LOG_FILE"

  echo "Published haiku to: $final_file" >&2
  echo "$final_file"
}

# ── Pipeline helper: run the full pipeline once ──────────────────────────────
#
# Usage:
#   run_haiku_pipeline [item_id]
#
# Runs fetch → generate → publish as a single pipeline invocation.
#
run_haiku_pipeline() {
  local item_id="${1:-}"
  local timestamp
  timestamp=$(date -u +%Y%m%d-%H%M%S)

  echo "[haiku-core] Starting pipeline at ${timestamp}" >&2

  # Step 1: Fetch news
  echo "[haiku-core] Fetching news..." >&2
  local news_json
  news_json=$(fetch_news "$item_id") || {
    echo "[haiku-core] ERROR: fetch_news failed" >&2
    return 1
  }

  local title
  title=$(echo "$news_json" | jq -r '.title // "unknown"' 2>/dev/null)
  echo "[haiku-core] Got news: ${title}" >&2

  # Step 2: Generate haiku
  echo "[haiku-core] Generating haiku..." >&2
  local haiku_text
  haiku_text=$(generate_haiku "$news_json") || {
    echo "[haiku-core] ERROR: generate_haiku failed" >&2
    return 1
  }
  echo "[haiku-core] Generated haiku:" >&2
  echo "$haiku_text" >&2

  # Step 3: Publish haiku
  echo "[haiku-core] Publishing haiku..." >&2
  local output_file
  output_file=$(publish_haiku "$haiku_text" "$news_json" "$timestamp") || {
    echo "[haiku-core] ERROR: publish_haiku failed" >&2
    return 1
  }

  echo "[haiku-core] Pipeline complete: ${output_file}" >&2
}

# ── Direct invocation support ─────────────────────────────────────────────────

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    run)
      shift
      run_haiku_pipeline "$@"
      ;;
    fetch)
      shift
      fetch_news "$@"
      ;;
    generate)
      shift
      generate_haiku "$@"
      ;;
    publish)
      shift
      publish_haiku "$@"
      ;;
    *)
      echo "Usage: $0 {run|fetch|generate|publish} [args...]"
      echo ""
      echo "Commands:"
      echo "  run [item_id]                        Run full pipeline (fetch→generate→publish)"
      echo "  fetch [item_id]                      Fetch a news item (JSON output)"
      echo "  generate <news_json>                 Generate haiku from news JSON"
      echo "  publish <haiku_text> <news_json> [ts] Publish haiku to stream"
      echo ""
      echo "Or source this file for function access:"
      echo "  source scripts/haiku-core.sh"
      echo "  news=\$(fetch_news)"
      echo "  haiku=\$(generate_haiku \"\$news\")"
      echo "  publish_haiku \"\$haiku\" \"\$news\""
      exit 1
      ;;
  esac
fi
