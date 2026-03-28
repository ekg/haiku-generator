"""
haiku_generator.py — Core haiku pipeline for the autohaiku system (Python edition).

Provides three pipeline functions:
  - generate_haiku  — Transform a news item into a haiku using an LLM
  - publish_haiku   — Save the haiku to the haiku stream (haikus/ dir + log)

Also provides a convenience function to run the full pipeline:
  - run_haiku_pipeline

Configuration is via environment variables:
  HAIKU_DIR       — Output directory (default: ./haikus)
  HAIKU_LOG_FILE  — Aggregate log file (default: ./haiku-log.txt)

Compatible with the bash version in scripts/haiku-core.sh.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional


# ── Configuration defaults ────────────────────────────────────────────────────

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HAIKU_DIR = os.environ.get("HAIKU_DIR", os.path.join(_PROJECT_DIR, "haikus"))
HAIKU_LOG_FILE = os.environ.get(
    "HAIKU_LOG_FILE", os.path.join(_PROJECT_DIR, "haiku-log.txt")
)


# ── generate_haiku ────────────────────────────────────────────────────────────


def generate_haiku(news_item: str) -> str:
    """
    Transforms a news item into a haiku using an LLM.

    Args:
        news_item: The news item to transform.

    Returns:
        A haiku based on the news item (three lines, 5-7-5 syllables).
    """
    prompt = (
        "Convert the following news item into a haiku (5-7-5 syllables).\n"
        f'News Item: "{news_item}"\n'
        "Haiku:"
    )

    try:
        # Placeholder for LLM API call
        # In a real implementation, you would:
        # 1. Initialize your LLM client (e.g., OpenAI, Anthropic, etc.)
        # 2. Make a call to the LLM API with the prompt.
        # Example using a hypothetical client:
        # response = client.chat.completions.create(
        #     model="openrouter/auto",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=50,
        #     temperature=0.7,
        # )
        # haiku = response.choices[0].message.content.strip()

        # For now, return a mock haiku
        haiku = "A news item's fate,\nTransformed to seventeen sounds,\nNature's gentle sigh."

        # Basic validation: check if it looks like a haiku structure (line breaks)
        if not isinstance(haiku, str) or "\n" not in haiku:
            return "A haiku could not be formed."

        return haiku

    except Exception as e:
        print(f"Error generating haiku: {e}", file=sys.stderr)
        return "A haiku could not be formed due to an error."


# ── publish_haiku ─────────────────────────────────────────────────────────────


def publish_haiku(
    haiku_text: str,
    news_json: Optional[str] = None,
    timestamp: Optional[str] = None,
    observer: str = "news",
    haiku_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> str:
    """
    Save a generated haiku to the haiku stream (file + log).

    This function handles data persistence for haikus by:
    1. Writing a .haiku file with YAML frontmatter to the haikus/ directory
       using an atomic write pattern (temp file + rename) for crash safety.
    2. Appending a human-readable entry to the aggregate haiku log file.

    Args:
        haiku_text: The haiku text (typically three lines, 5-7-5 syllables).
        news_json:  Optional JSON string with news metadata (title, url, etc.)
                    used for frontmatter in the .haiku file.
        timestamp:  Optional timestamp string in YYYYMMDD-HHMMSS format.
                    Defaults to current UTC time.
        observer:   Observer name for the filename prefix (default: "news").
        haiku_dir:  Override the output directory (default: HAIKU_DIR).
        log_file:   Override the log file path (default: HAIKU_LOG_FILE).

    Returns:
        The path to the written .haiku file, or an empty string if haiku_text
        was empty.

    Raises:
        OSError: If file operations fail (disk full, permissions, etc.)

    Example:
        >>> path = publish_haiku(
        ...     "Markets rise today\\nBulls charge through the trading floor\\nGreen across the board",
        ...     '{"title": "Stock Rally", "url": "https://example.com/rally"}',
        ... )
        >>> print(path)
        /home/erik/autohaiku/haikus/news-20260327-183000.haiku
    """
    if not haiku_text or not haiku_text.strip():
        return ""

    # Resolve configuration
    out_dir = haiku_dir or HAIKU_DIR
    out_log = log_file or HAIKU_LOG_FILE

    # Generate timestamp if not provided
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Parse news metadata for frontmatter
    title = ""
    source_url = ""
    if news_json:
        try:
            obj = json.loads(news_json) if isinstance(news_json, str) else news_json
            title = obj.get("title") or obj.get("headline") or obj.get("subject") or ""
            source_url = obj.get("url") or obj.get("link") or obj.get("source_url") or ""
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    # Build file paths
    final_file = os.path.join(out_dir, f"{observer}-{timestamp}.haiku")
    tmp_file = os.path.join(out_dir, f".{observer}-{timestamp}.haiku.tmp")

    # ── Step 1: Atomic write to .haiku file ──────────────────────────────
    # Write to a temp file first, then rename. This ensures:
    #   - No partial .haiku files if the process is interrupted
    #   - os.rename() on the same filesystem is atomic (single rename(2) syscall)

    haiku_content_lines = [
        "---\n",
        f"observer: {observer}\n",
        f"timestamp: {timestamp}\n",
    ]
    if title:
        haiku_content_lines.append(f"source_title: {title}\n")
    if source_url:
        haiku_content_lines.append(f"source_url: {source_url}\n")
    haiku_content_lines.append("---\n")
    haiku_content_lines.append(haiku_text.strip() + "\n")

    with open(tmp_file, "w", encoding="utf-8") as f:
        f.writelines(haiku_content_lines)

    # Atomic rename (same filesystem guarantees atomicity)
    os.rename(tmp_file, final_file)

    # ── Step 2: Append to aggregate log ──────────────────────────────────
    # Format the timestamp for human readability
    human_ts = timestamp
    try:
        # Convert YYYYMMDD-HHMMSS → YYYY-MM-DDTHH:MM:SS
        if len(timestamp) == 15 and "-" in timestamp:
            dt_part, tm_part = timestamp.split("-", 1)
            human_ts = (
                f"{dt_part[:4]}-{dt_part[4:6]}-{dt_part[6:8]}"
                f"T{tm_part[:2]}:{tm_part[2:4]}:{tm_part[4:6]}"
            )
    except (ValueError, IndexError):
        pass  # keep the raw timestamp

    log_entry = (
        f"=== {human_ts} ===\n"
        f"[{observer}] {title}\n"
        f"{haiku_text.strip()}\n"
        "\n"
        "===\n"
        "\n"
    )

    with open(out_log, "a", encoding="utf-8") as log:
        log.write(log_entry)

    print(f"Published haiku to: {final_file}", file=sys.stderr)
    return final_file


# ── run_haiku_pipeline ────────────────────────────────────────────────────────


def run_haiku_pipeline(news_item: Optional[str] = None) -> str:
    """
    Run the full haiku pipeline: generate → publish.

    Args:
        news_item: A news item string. If None, uses a default example.

    Returns:
        The path to the published .haiku file.
    """
    if news_item is None:
        news_item = "A quiet day in the world of technology."

    haiku_text = generate_haiku(news_item)
    news_json = json.dumps({"title": news_item[:80]})
    return publish_haiku(haiku_text, news_json)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "publish":
        # Usage: python haiku_generator.py publish "haiku text" '{"title":"..."}' [timestamp]
        if len(sys.argv) < 3:
            print("Usage: python haiku_generator.py publish <haiku_text> [news_json] [timestamp]", file=sys.stderr)
            sys.exit(1)
        h_text = sys.argv[2]
        n_json = sys.argv[3] if len(sys.argv) > 3 else None
        ts = sys.argv[4] if len(sys.argv) > 4 else None
        result = publish_haiku(h_text, n_json, ts)
        print(result)
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        item = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        result = run_haiku_pipeline(item)
        print(result)
    else:
        # Default: example usage
        example_news = "The stock market experienced a significant rally today, with major indices closing up by over 2%."
        haiku = generate_haiku(example_news)
        print(f"News Item: {example_news}")
        print(f"Haiku:\n{haiku}")

        print("\nPublishing haiku...")
        news_json = json.dumps({"title": example_news[:80], "url": "https://example.com/rally"})
        path = publish_haiku(haiku, news_json)
        print(f"Published to: {path}")
