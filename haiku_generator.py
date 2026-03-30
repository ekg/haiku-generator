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
import hashlib
import glob
import threading
from datetime import datetime, timezone
from typing import Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


# ── Configuration defaults ────────────────────────────────────────────────────

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HAIKU_DIR = os.environ.get("HAIKU_DIR", os.path.join(_PROJECT_DIR, "haikus"))
HAIKU_LOG_FILE = os.environ.get(
    "HAIKU_LOG_FILE", os.path.join(_PROJECT_DIR, "haiku-log.txt")
)


# ── Duplicate Detection ───────────────────────────────────────────────────────


def _normalize_haiku_text(haiku_text: str) -> str:
    """
    Normalize haiku text for duplicate comparison by:
    - Stripping whitespace
    - Converting to lowercase
    - Normalizing line endings
    - Removing extra spaces
    """
    return "\n".join(
        line.strip().lower()
        for line in haiku_text.strip().split("\n")
        if line.strip()
    )


def _haiku_content_hash(haiku_text: str) -> str:
    """Generate a content hash for a haiku for duplicate detection."""
    normalized = _normalize_haiku_text(haiku_text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def _extract_haiku_content(haiku_file_path: str) -> str:
    """Extract the haiku content (after frontmatter) from a .haiku file."""
    try:
        with open(haiku_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by YAML frontmatter delimiters
        parts = content.split('---')
        if len(parts) >= 3:
            # Content after second '---'
            haiku_content = '---'.join(parts[2:]).strip()
            return haiku_content
        else:
            # No frontmatter, return whole content
            return content.strip()
    except (OSError, UnicodeDecodeError):
        return ""


def check_duplicate_haiku(haiku_text: str, haiku_dir: Optional[str] = None) -> Optional[str]:
    """
    Check if a haiku already exists in the haiku directory.

    Args:
        haiku_text: The haiku text to check
        haiku_dir: Directory to search (defaults to HAIKU_DIR)

    Returns:
        Path to existing haiku file if duplicate found, None otherwise
    """
    search_dir = haiku_dir or HAIKU_DIR
    if not os.path.exists(search_dir):
        return None

    target_hash = _haiku_content_hash(haiku_text)

    # Check all existing .haiku files
    for haiku_file in glob.glob(os.path.join(search_dir, "*.haiku")):
        existing_content = _extract_haiku_content(haiku_file)
        if existing_content and _haiku_content_hash(existing_content) == target_hash:
            return haiku_file

    return None


# ── File System Monitoring ────────────────────────────────────────────────────


class HaikuDirectoryHandler(FileSystemEventHandler):
    """File system event handler for monitoring the haikus directory."""

    def __init__(self, log_file: Optional[str] = None):
        super().__init__()
        self.log_file = log_file or HAIKU_LOG_FILE
        self.monitored_extensions = {'.haiku'}

    def _log_event(self, event_type: str, path: str) -> None:
        """Log file system events to the monitoring log."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        log_entry = f"[{timestamp}] [FS-MONITOR] {event_type}: {os.path.basename(path)}\n"

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except OSError:
            print(f"Warning: Could not write to monitoring log: {self.log_file}", file=sys.stderr)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in self.monitored_extensions):
            self._log_event("CREATED", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in self.monitored_extensions):
            self._log_event("MODIFIED", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in self.monitored_extensions):
            self._log_event("DELETED", event.src_path)


class HaikuDirectoryMonitor:
    """Manages file system monitoring for the haikus directory."""

    def __init__(self, haiku_dir: Optional[str] = None, log_file: Optional[str] = None):
        self.haiku_dir = haiku_dir or HAIKU_DIR
        self.log_file = log_file or HAIKU_LOG_FILE
        self.observer = None
        self.handler = None
        self._lock = threading.Lock()

    def start_monitoring(self) -> bool:
        """
        Start monitoring the haikus directory for file system changes.

        Returns:
            True if monitoring started successfully, False otherwise
        """
        with self._lock:
            if self.observer and self.observer.is_alive():
                return True  # Already monitoring

            if not os.path.exists(self.haiku_dir):
                os.makedirs(self.haiku_dir, exist_ok=True)

            try:
                self.handler = HaikuDirectoryHandler(self.log_file)
                self.observer = Observer()
                self.observer.schedule(self.handler, self.haiku_dir, recursive=False)
                self.observer.start()

                # Log that monitoring started
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                log_entry = f"[{timestamp}] [FS-MONITOR] STARTED: Monitoring {self.haiku_dir}\n"
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)

                return True
            except Exception as e:
                print(f"Error starting file system monitor: {e}", file=sys.stderr)
                return False

    def stop_monitoring(self) -> None:
        """Stop monitoring the haikus directory."""
        with self._lock:
            if self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join(timeout=5)

                # Log that monitoring stopped
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                log_entry = f"[{timestamp}] [FS-MONITOR] STOPPED: Monitoring {self.haiku_dir}\n"
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
                except OSError:
                    pass

    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        with self._lock:
            return self.observer is not None and self.observer.is_alive()


# Global monitor instance
_global_monitor = None


def start_file_system_monitoring(haiku_dir: Optional[str] = None, log_file: Optional[str] = None) -> bool:
    """
    Start file system monitoring for the haikus directory.

    Returns:
        True if monitoring started successfully, False otherwise
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HaikuDirectoryMonitor(haiku_dir, log_file)
    return _global_monitor.start_monitoring()


def stop_file_system_monitoring() -> None:
    """Stop file system monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


def is_monitoring_active() -> bool:
    """Check if file system monitoring is currently active."""
    global _global_monitor
    return _global_monitor is not None and _global_monitor.is_monitoring()


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
    skip_duplicate_check: bool = False,
) -> str:
    """
    Save a generated haiku to the haiku stream (file + log).

    This function handles data persistence for haikus by:
    1. Checking for duplicate content (unless skip_duplicate_check=True)
    2. Writing a .haiku file with YAML frontmatter to the haikus/ directory
       using an atomic write pattern (temp file + rename) for crash safety.
    3. Appending a human-readable entry to the aggregate haiku log file.

    Args:
        haiku_text: The haiku text (typically three lines, 5-7-5 syllables).
        news_json:  Optional JSON string with news metadata (title, url, etc.)
                    used for frontmatter in the .haiku file.
        timestamp:  Optional timestamp string in YYYYMMDD-HHMMSS format.
                    Defaults to current UTC time.
        observer:   Observer name for the filename prefix (default: "news").
        haiku_dir:  Override the output directory (default: HAIKU_DIR).
        log_file:   Override the log file path (default: HAIKU_LOG_FILE).
        skip_duplicate_check: If True, skip duplicate detection (default: False).

    Returns:
        The path to the written .haiku file, or an empty string if haiku_text
        was empty or if a duplicate was found.

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

    # Check for duplicate content (unless explicitly skipped)
    if not skip_duplicate_check:
        existing_file = check_duplicate_haiku(haiku_text, out_dir)
        if existing_file:
            print(f"Duplicate haiku detected, skipping: {os.path.basename(existing_file)}", file=sys.stderr)
            return ""

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
    elif len(sys.argv) > 1 and sys.argv[1] == "check-duplicate":
        # Usage: python haiku_generator.py check-duplicate "haiku text"
        if len(sys.argv) < 3:
            print("Usage: python haiku_generator.py check-duplicate <haiku_text>", file=sys.stderr)
            sys.exit(1)
        h_text = sys.argv[2]
        duplicate_file = check_duplicate_haiku(h_text)
        if duplicate_file:
            print(f"Duplicate found: {duplicate_file}")
            sys.exit(1)
        else:
            print("No duplicate found")
            sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "start-monitoring":
        # Usage: python haiku_generator.py start-monitoring
        success = start_file_system_monitoring()
        if success:
            print("File system monitoring started")
            # Keep running until interrupted
            try:
                while is_monitoring_active():
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                stop_file_system_monitoring()
        else:
            print("Failed to start file system monitoring", file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "stop-monitoring":
        # Usage: python haiku_generator.py stop-monitoring
        stop_file_system_monitoring()
        print("File system monitoring stopped")
    elif len(sys.argv) > 1 and sys.argv[1] == "monitoring-status":
        # Usage: python haiku_generator.py monitoring-status
        if is_monitoring_active():
            print("File system monitoring is ACTIVE")
        else:
            print("File system monitoring is INACTIVE")
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
