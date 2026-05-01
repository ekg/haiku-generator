"""CPU-only character/control-token n-gram haiku baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

from local_haiku_dataset import normalize_poem_text
from local_haiku_tokenizer import (
    END_TOKEN,
    HAIKU_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    PROMPT_END_TOKEN,
    PROMPT_TOKEN,
    build_training_examples,
    decode_tokens,
    iter_dataset_records,
)


DEFAULT_ORDER = 4
DEFAULT_ALPHA = 0.05
DEFAULT_MAX_LINE_CHARS = (32, 44, 32)
DEFAULT_MIN_LINE_CHARS = (8, 12, 8)
CONTROL_TOKENS = {
    HAIKU_TOKEN,
    PROMPT_TOKEN,
    PROMPT_END_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    END_TOKEN,
}


@dataclass(frozen=True)
class NGramModel:
    """Smoothed token n-gram counts plus corpus rejection metadata."""

    order: int
    effective_order: int
    alpha: float
    counts: dict[int, dict[tuple[str, ...], Counter[str]]]
    vocabulary: tuple[str, ...]
    train_poem_hashes: frozenset[str]
    train_poem_count: int
    metadata: dict[str, object]

    @property
    def context_size(self) -> int:
        return max(0, self.effective_order - 1)


def train_model(
    dataset_path: str | Path,
    *,
    order: int = DEFAULT_ORDER,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[NGramModel, dict[str, object]]:
    """Train n-gram transition counts from the training split only."""
    if order < 1:
        raise ValueError("order must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    records = list(iter_dataset_records(dataset_path))
    train_build = build_training_examples(records, allowed_splits={"train"})
    if not train_build.examples:
        raise ValueError("dataset contains no valid train examples")

    train_streams = [list(example["tokens"]) for example in train_build.examples]
    longest_stream = max(len(stream) for stream in train_streams)
    effective_order = min(order, max(1, longest_stream))
    counts = _count_streams(train_streams, effective_order)
    vocabulary = tuple(sorted({token for stream in train_streams for token in stream}, key=_token_sort_key))

    train_poems = [
        str(record.get("text") or "\n".join(str(line) for line in record.get("lines", [])))
        for record in records
        if str(record.get("split")) == "train"
    ]
    train_poem_hashes = frozenset(_poem_hash(poem) for poem in train_poems if poem.strip())
    dev_metrics = _split_likelihood(records, {"dev"}, counts, vocabulary, effective_order, alpha)

    model = NGramModel(
        order=order,
        effective_order=effective_order,
        alpha=alpha,
        counts=counts,
        vocabulary=vocabulary,
        train_poem_hashes=train_poem_hashes,
        train_poem_count=len(train_poem_hashes),
        metadata={
            "format": "local-haiku-ngram-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "cpu_only": True,
            "smoothing": "add-alpha backoff over learned n-gram counts",
        },
    )
    metrics = {
        "format": "local-haiku-ngram-metrics-v1",
        "dataset": str(dataset_path),
        "requested_order": order,
        "effective_order": effective_order,
        "alpha": alpha,
        "train_example_count": len(train_build.examples),
        "skipped_invalid_train_count": train_build.skipped_invalid_count,
        "vocabulary_size": len(vocabulary),
        "transition_context_count": sum(len(contexts) for contexts in counts.values()),
        "train_poem_hash_count": len(train_poem_hashes),
        **dev_metrics,
        "runtime": "CPU-only Python standard library; no cloud APIs or GPU dependencies",
    }
    return model, metrics


def generate_haiku(
    model: NGramModel,
    *,
    prompt: str = "",
    observer: str | None = None,
    seed: int | None = None,
    max_attempts: int = 200,
) -> tuple[str, dict[str, object]]:
    """Generate one accepted three-line haiku candidate."""
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        tokens = _generate_tokens(model, prompt=prompt, observer=observer, rng=rng)
        try:
            decoded = decode_tokens(tokens)
        except ValueError:
            continue
        text = decoded.text
        rejection = rejection_reason(model, text)
        if rejection is None:
            return text, {
                "attempt": attempt,
                "prompt": prompt,
                "observer": observer or _observer_from_prompt(prompt),
                "seed": seed,
                "tokens": tokens,
            }

    raise RuntimeError(f"no accepted candidate after {max_attempts} attempts")


def rejection_reason(model: NGramModel, text: str) -> str | None:
    """Return a candidate rejection reason, or None if accepted."""
    lines = tuple(line.strip() for line in text.split("\n"))
    if len(lines) != 3 or any(not line for line in lines):
        return "not_three_nonempty_lines"
    if len(set(line.casefold() for line in lines)) != 3:
        return "repeated_full_line"
    if any(len(line) > limit for line, limit in zip(lines, DEFAULT_MAX_LINE_CHARS)):
        return "line_too_long"
    if _poem_hash(text) in model.train_poem_hashes:
        return "exact_train_duplicate"
    return None


def save_model(model: NGramModel, path: str | Path) -> None:
    """Write a compact gzipped JSON model artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "local-haiku-ngram-v1",
        "order": model.order,
        "effective_order": model.effective_order,
        "alpha": model.alpha,
        "vocabulary": list(model.vocabulary),
        "train_poem_hashes": sorted(model.train_poem_hashes),
        "train_poem_count": model.train_poem_count,
        "metadata": model.metadata,
        "counts": {
            str(order): {
                _context_key(context): dict(counter)
                for context, counter in sorted(contexts.items(), key=lambda item: item[0])
            }
            for order, contexts in sorted(model.counts.items())
        },
    }
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def load_model(path: str | Path) -> NGramModel:
    """Load a gzipped JSON model artifact."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != "local-haiku-ngram-v1":
        raise ValueError(f"unsupported model format: {payload.get('format')!r}")
    counts = {
        int(order): {
            tuple(json.loads(context_key)): Counter(counter)
            for context_key, counter in contexts.items()
        }
        for order, contexts in payload["counts"].items()
    }
    return NGramModel(
        order=int(payload["order"]),
        effective_order=int(payload["effective_order"]),
        alpha=float(payload["alpha"]),
        counts=counts,
        vocabulary=tuple(payload["vocabulary"]),
        train_poem_hashes=frozenset(payload.get("train_poem_hashes", [])),
        train_poem_count=int(payload.get("train_poem_count", 0)),
        metadata=dict(payload.get("metadata", {})),
    )


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CPU-only local haiku n-gram model.")
    parser.add_argument("--dataset", required=True, help="Normalized local haiku dataset JSONL.")
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER, help="Requested n-gram order.")
    parser.add_argument("--out", required=True, help="Write model JSON gzip artifact.")
    parser.add_argument("--metrics", help="Optional metrics JSON output path.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Add-alpha smoothing value.")
    args = parser.parse_args(argv)

    model, metrics = train_model(args.dataset, order=args.order, alpha=args.alpha)
    save_model(model, args.out)
    if args.metrics:
        metrics_out = Path(args.metrics)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dev = metrics.get("dev_perplexity")
    dev_text = "n/a" if dev is None else f"{float(dev):.3f}"
    print(
        f"Wrote CPU-only n-gram model to {args.out}; "
        f"train_examples={metrics['train_example_count']} "
        f"effective_order={metrics['effective_order']} "
        f"vocabulary_size={metrics['vocabulary_size']} "
        f"dev_perplexity={dev_text}"
    )
    return 0


def generate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate from a local CPU-only haiku n-gram model.")
    parser.add_argument("--model", required=True, help="Model JSON gzip artifact.")
    parser.add_argument("--prompt", default="", help="Prompt text to condition the control-token prefix.")
    parser.add_argument("--observer", help="Observer tag override, for example disk/network/process.")
    parser.add_argument("--seed", type=int, help="Deterministic random seed.")
    parser.add_argument("--max-attempts", type=int, default=200, help="Candidate retry budget.")
    parser.add_argument("--samples-out", help="Optional JSONL file for the accepted sample.")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    poem, metadata = generate_haiku(
        model,
        prompt=args.prompt,
        observer=args.observer,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    if args.samples_out:
        out = Path(args.samples_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"prompt": args.prompt, "poem": poem, "lines": poem.split("\n"), "metadata": metadata},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
    print(poem)
    return 0


def _count_streams(
    streams: Sequence[Sequence[str]], effective_order: int
) -> dict[int, dict[tuple[str, ...], Counter[str]]]:
    counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
        order: defaultdict(Counter) for order in range(1, effective_order + 1)
    }
    for stream in streams:
        for index, token in enumerate(stream):
            for order in range(1, effective_order + 1):
                context_size = order - 1
                if index < context_size:
                    continue
                context = tuple(stream[index - context_size : index])
                counts[order][context][token] += 1
    return {order: dict(contexts) for order, contexts in counts.items()}


def _split_likelihood(
    records: Sequence[dict[str, object]],
    splits: set[str],
    counts: Mapping[int, Mapping[tuple[str, ...], Counter[str]]],
    vocabulary: Sequence[str],
    effective_order: int,
    alpha: float,
) -> dict[str, object]:
    build = build_training_examples(records, allowed_splits=splits)
    token_count = 0
    nll = 0.0
    for example in build.examples:
        stream = list(example["tokens"])
        for index, token in enumerate(stream):
            context = tuple(stream[max(0, index - effective_order + 1) : index])
            probability = _probability(token, context, counts, vocabulary, effective_order, alpha)
            nll -= math.log(max(probability, 1e-12))
            token_count += 1
    if token_count == 0:
        return {
            "dev_example_count": 0,
            "dev_token_count": 0,
            "dev_negative_log_likelihood": None,
            "dev_perplexity": None,
        }
    avg_nll = nll / token_count
    return {
        "dev_example_count": len(build.examples),
        "dev_token_count": token_count,
        "dev_negative_log_likelihood": avg_nll,
        "dev_perplexity": math.exp(avg_nll),
    }


def _probability(
    token: str,
    context: tuple[str, ...],
    counts: Mapping[int, Mapping[tuple[str, ...], Counter[str]]],
    vocabulary: Sequence[str],
    effective_order: int,
    alpha: float,
) -> float:
    used_context = context[-(effective_order - 1) :] if effective_order > 1 else ()
    for order in range(min(effective_order, len(used_context) + 1), 0, -1):
        prefix = used_context[-(order - 1) :] if order > 1 else ()
        counter = counts.get(order, {}).get(prefix)
        if counter:
            total = sum(counter.values())
            return (counter.get(token, 0) + alpha) / (total + alpha * len(vocabulary))
    return 1.0 / max(1, len(vocabulary))


def _generate_tokens(
    model: NGramModel,
    *,
    prompt: str,
    observer: str | None,
    rng: random.Random,
) -> list[str]:
    observer_value = observer or _observer_from_prompt(prompt)
    tokens = [
        HAIKU_TOKEN,
        "<LANG=en>",
        _metadata_token("OBSERVER", observer_value),
        "<SOURCE=repo-local>",
        PROMPT_TOKEN,
        *list(prompt),
        PROMPT_END_TOKEN,
        L1_TOKEN,
    ]
    line_index = 0
    line_chars = [0, 0, 0]

    for _ in range(sum(DEFAULT_MAX_LINE_CHARS) + 12):
        allowed = _allowed_tokens(model, line_index, line_chars[line_index])
        token = _sample_next(model, tokens, allowed, rng)
        tokens.append(token)
        if token in {L2_TOKEN, L3_TOKEN, END_TOKEN}:
            if token == END_TOKEN:
                return tokens
            line_index += 1
            if line_index > 2:
                tokens.append(END_TOKEN)
                return tokens
            continue
        line_chars[line_index] += 1

    if tokens[-1] != END_TOKEN:
        tokens.append(END_TOKEN)
    return tokens


def _allowed_tokens(model: NGramModel, line_index: int, current_length: int) -> set[str]:
    boundary = (L2_TOKEN, L3_TOKEN, END_TOKEN)[line_index]
    min_len = DEFAULT_MIN_LINE_CHARS[line_index]
    max_len = DEFAULT_MAX_LINE_CHARS[line_index]
    allowed = {
        token
        for token in model.vocabulary
        if token not in CONTROL_TOKENS and not _is_metadata_token(token) and len(token) == 1 and token != "\n"
    }
    if current_length >= min_len:
        allowed.add(boundary)
    if current_length >= max_len:
        return {boundary}
    return allowed


def _sample_next(
    model: NGramModel,
    tokens: Sequence[str],
    allowed: set[str],
    rng: random.Random,
) -> str:
    weights: Counter[str] = Counter()
    for order in range(min(model.effective_order, len(tokens) + 1), 0, -1):
        context_size = order - 1
        context = tuple(tokens[-context_size:]) if context_size else ()
        counter = model.counts.get(order, {}).get(context)
        if not counter:
            continue
        backoff_weight = order * order
        for token, count in counter.items():
            if token in allowed:
                weights[token] += count * backoff_weight
        if weights:
            break

    if not weights:
        unigram = model.counts.get(1, {}).get((), Counter())
        for token in allowed:
            if unigram.get(token, 0):
                weights[token] += unigram[token]
    if not weights:
        return sorted(allowed)[0]
    total = float(sum(weights.values()))
    threshold = rng.random() * total
    running = 0.0
    for token, weight in sorted(weights.items(), key=lambda item: item[0]):
        running += weight
        if running >= threshold:
            return token
    return next(reversed(sorted(weights)))


def _observer_from_prompt(prompt: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", prompt.casefold()))
    if words & {"disk", "drive", "filesystem", "file", "bytes", "root"}:
        return "disk"
    if words & {"network", "localhost", "latency", "packet", "packets", "ping", "loopback"}:
        return "network"
    if words & {"process", "cpu", "load", "memory", "thread"}:
        return "process"
    return "unknown"


def _poem_hash(text: str) -> str:
    normalized = normalize_poem_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _context_key(context: tuple[str, ...]) -> str:
    return json.dumps(list(context), ensure_ascii=False, separators=(",", ":"))


def _metadata_token(key: str, value: str) -> str:
    return f"<{key}={quote(value, safe='._~-')}>"


def _is_metadata_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">") and "=" in token


def _token_sort_key(token: str) -> tuple[int, str]:
    if token in CONTROL_TOKENS:
        return (0, token)
    if _is_metadata_token(token):
        return (1, token)
    return (2, token)


if __name__ == "__main__":
    raise SystemExit(train_main())
