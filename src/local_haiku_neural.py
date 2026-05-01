"""Tiny local GRU character/control-token haiku language model.

The default implementation is a CPU NumPy GRU so training and inference remain
local even when PyTorch is absent. PyTorch/ROCm can be used by later follow-up
work as optional acceleration, but it is not required for this checkpoint
format or command path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Mapping, Sequence
from urllib.parse import quote

import numpy as np

from local_haiku_dataset import normalize_poem_text
from local_haiku_ngram import (
    DEFAULT_MAX_LINE_CHARS,
    DEFAULT_MIN_LINE_CHARS,
    PROMPT_TOPIC_ANCHORS,
)
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


FORMAT = "local-haiku-neural-gru-v1"
METRICS_FORMAT = "local-haiku-neural-gru-metrics-v1"
UNK_TOKEN = "<UNK>"
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_LAYERS = 1
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_SEED = 1
CONTROL_TOKENS = {
    HAIKU_TOKEN,
    PROMPT_TOKEN,
    PROMPT_END_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
}


@dataclass(frozen=True)
class NeuralHaikuModel:
    """Serializable GRU weights and vocabulary metadata."""

    vocabulary: tuple[str, ...]
    weights: dict[str, np.ndarray]
    embedding_dim: int
    hidden_size: int
    layers: int
    train_poem_hashes: frozenset[str]
    train_poem_count: int
    metadata: dict[str, object]

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}


def train_model(
    dataset_path: str | Path,
    *,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    layers: int = DEFAULT_LAYERS,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
) -> tuple[NeuralHaikuModel, dict[str, object]]:
    """Train a tiny CPU GRU language model over tokenizer streams."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if layers != 1:
        raise ValueError("this local NumPy checkpoint currently supports exactly one GRU layer")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    records = list(iter_dataset_records(dataset_path))
    train_build = build_training_examples(records, allowed_splits={"train"})
    if not train_build.examples:
        raise ValueError("dataset contains no valid train examples")

    train_streams = [list(example["tokens"]) for example in train_build.examples]
    vocabulary = _build_neural_vocabulary(train_streams)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    train_ids = [_stream_to_ids(stream, token_to_id) for stream in train_streams]

    rng = np.random.default_rng(seed)
    weights = _init_weights(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        rng=rng,
    )

    epoch_losses: list[float] = []
    transition_count = sum(max(0, len(stream) - 1) for stream in train_ids)
    for _epoch in range(epochs):
        order = rng.permutation(len(train_ids))
        total_loss = 0.0
        total_tokens = 0
        for stream_index in order:
            ids = train_ids[int(stream_index)]
            if len(ids) < 2:
                continue
            loss, grads, token_count = _loss_and_grads(weights, ids)
            _clip_grads(grads, max_norm=5.0)
            for name, grad in grads.items():
                weights[name] -= learning_rate * grad
            total_loss += loss
            total_tokens += token_count
        epoch_losses.append(total_loss / max(1, total_tokens))

    dev_metrics = _split_likelihood(records, {"dev"}, weights, vocabulary)
    train_poems = [
        str(record.get("text") or "\n".join(str(line) for line in record.get("lines", [])))
        for record in records
        if str(record.get("split")) == "train"
    ]
    train_poem_hashes = frozenset(_poem_hash(poem) for poem in train_poems if poem.strip())
    model = NeuralHaikuModel(
        vocabulary=vocabulary,
        weights=weights,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        layers=layers,
        train_poem_hashes=train_poem_hashes,
        train_poem_count=len(train_poem_hashes),
        metadata={
            "format": FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "backend": "numpy-gru-cpu",
            "cpu_default": True,
            "cloud_api_required": False,
            "optional_acceleration": "PyTorch CPU/ROCm may be used by later compatible trainers; GPU is not required.",
        },
    )
    metrics = {
        "format": METRICS_FORMAT,
        "dataset": str(dataset_path),
        "backend": "numpy-gru-cpu",
        "architecture": "character/control-token GRU language model",
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "layers": layers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_example_count": len(train_build.examples),
        "skipped_invalid_train_count": train_build.skipped_invalid_count,
        "train_transition_count": transition_count,
        "train_loss": epoch_losses[-1],
        "train_perplexity": math.exp(min(50.0, epoch_losses[-1])),
        "epoch_losses": epoch_losses,
        "vocabulary_size": len(vocabulary),
        "train_poem_hash_count": len(train_poem_hashes),
        **dev_metrics,
        "runtime": "Local CPU NumPy GRU; no cloud APIs. PyTorch/GPU acceleration is optional and not required.",
    }
    return model, metrics


def generate_haiku(
    model: NeuralHaikuModel,
    *,
    prompt: str = "",
    observer: str | None = None,
    seed: int | None = None,
    temperature: float = 0.9,
    max_attempts: int = 200,
) -> tuple[str, dict[str, object]]:
    """Generate one accepted three-line candidate from a neural checkpoint."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        tokens = _generate_tokens(model, prompt=prompt, observer=observer, temperature=temperature, rng=rng)
        try:
            decoded = decode_tokens(tokens)
        except ValueError:
            continue
        text = decoded.text
        rejection = rejection_reason(model, text)
        if rejection is None:
            rejection = _prompt_rejection_reason(prompt, text)
        if rejection is None:
            return text, {
                "attempt": attempt,
                "prompt": prompt,
                "observer": observer or _observer_from_prompt(prompt),
                "seed": seed,
                "temperature": temperature,
                "tokens": tokens,
            }

    raise RuntimeError(f"no accepted candidate after {max_attempts} attempts")


def rejection_reason(model: NeuralHaikuModel, text: str) -> str | None:
    """Return a candidate rejection reason, or None if accepted."""

    lines = tuple(line.strip() for line in text.split("\n"))
    if len(lines) != 3 or any(not line for line in lines):
        return "not_three_nonempty_lines"
    if len(set(line.casefold() for line in lines)) != 3:
        return "repeated_full_line"
    words = re.findall(r"[a-z0-9_./:%']+", " ".join(lines).casefold())
    phrases = [tuple(words[index : index + 3]) for index in range(0, max(0, len(words) - 2))]
    if len(phrases) != len(set(phrases)):
        return "repeated_phrase"
    if any(len(line) > limit for line, limit in zip(lines, DEFAULT_MAX_LINE_CHARS)):
        return "line_too_long"
    if _poem_hash(text) in model.train_poem_hashes:
        return "exact_train_duplicate"
    return None


def save_model(model: NeuralHaikuModel, checkpoint_path: str | Path, metadata_path: str | Path) -> None:
    """Write NumPy checkpoint weights plus JSON metadata."""

    checkpoint = Path(checkpoint_path)
    metadata = Path(metadata_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, **model.weights)
    checkpoint_ref = _checkpoint_reference(checkpoint, metadata.parent)
    payload = {
        "format": FORMAT,
        "checkpoint": checkpoint_ref,
        "vocabulary": list(model.vocabulary),
        "embedding_dim": model.embedding_dim,
        "hidden_size": model.hidden_size,
        "layers": model.layers,
        "train_poem_hashes": sorted(model.train_poem_hashes),
        "train_poem_count": model.train_poem_count,
        "metadata": model.metadata,
    }
    metadata.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_model(metadata_path: str | Path) -> NeuralHaikuModel:
    """Load a neural checkpoint from its JSON metadata path."""

    metadata_file = Path(metadata_path)
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported model format: {payload.get('format')!r}")
    checkpoint = Path(payload["checkpoint"])
    if not checkpoint.is_absolute():
        metadata_relative = metadata_file.parent / checkpoint
        checkpoint = metadata_relative if metadata_relative.exists() else checkpoint
    with np.load(checkpoint) as loaded:
        weights = {name: loaded[name].astype(np.float64) for name in loaded.files}
    return NeuralHaikuModel(
        vocabulary=tuple(payload["vocabulary"]),
        weights=weights,
        embedding_dim=int(payload["embedding_dim"]),
        hidden_size=int(payload["hidden_size"]),
        layers=int(payload["layers"]),
        train_poem_hashes=frozenset(payload.get("train_poem_hashes", [])),
        train_poem_count=int(payload.get("train_poem_count", 0)),
        metadata=dict(payload.get("metadata", {})),
    )


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the local CPU tiny-GRU haiku model.")
    parser.add_argument("--dataset", required=True, help="Normalized local haiku dataset JSONL.")
    parser.add_argument("--out", required=True, help="Write NumPy checkpoint .npz artifact.")
    parser.add_argument("--metadata", required=True, help="Write JSON checkpoint metadata.")
    parser.add_argument("--metrics", help="Optional metrics JSON output path.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    model, metrics = train_model(
        args.dataset,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    save_model(model, args.out, args.metadata)
    if args.metrics:
        metrics_out = Path(args.metrics)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dev = metrics.get("dev_perplexity")
    dev_text = "n/a" if dev is None else f"{float(dev):.3f}"
    print(
        f"Wrote local GRU checkpoint to {args.out}; "
        f"train_examples={metrics['train_example_count']} "
        f"vocabulary_size={metrics['vocabulary_size']} "
        f"train_loss={metrics['train_loss']:.4f} "
        f"dev_perplexity={dev_text}"
    )
    return 0


def generate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate from a local CPU tiny-GRU haiku model.")
    parser.add_argument("--model", required=True, help="Model metadata JSON artifact.")
    parser.add_argument("--prompt", default="", help="Prompt text to condition the control-token prefix.")
    parser.add_argument("--observer", help="Observer tag override, for example disk/network/process.")
    parser.add_argument("--seed", type=int, help="Deterministic random seed.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-attempts", type=int, default=200, help="Candidate retry budget.")
    parser.add_argument("--samples-out", help="Optional JSONL file for the accepted sample.")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    poem, metadata = generate_haiku(
        model,
        prompt=args.prompt,
        observer=args.observer,
        seed=args.seed,
        temperature=args.temperature,
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


def _loss_and_grads(weights: Mapping[str, np.ndarray], ids: Sequence[int]) -> tuple[float, dict[str, np.ndarray], int]:
    grads = {name: np.zeros_like(value) for name, value in weights.items()}
    h = np.zeros(weights["b_z"].shape[0])
    cache = []
    total_loss = 0.0
    token_count = 0
    for input_id, target_id in zip(ids[:-1], ids[1:]):
        e = weights["E"][input_id]
        h_prev = h
        z = _sigmoid(e @ weights["W_z"] + h_prev @ weights["U_z"] + weights["b_z"])
        r = _sigmoid(e @ weights["W_r"] + h_prev @ weights["U_r"] + weights["b_r"])
        g = np.tanh(e @ weights["W_h"] + (r * h_prev) @ weights["U_h"] + weights["b_h"])
        h = (1.0 - z) * h_prev + z * g
        logits = h @ weights["W_o"] + weights["b_o"]
        probs = _softmax(logits)
        total_loss -= math.log(max(float(probs[target_id]), 1e-12))
        cache.append((input_id, target_id, e, h_prev, z, r, g, h, probs))
        token_count += 1

    dh_next = np.zeros_like(h)
    for input_id, target_id, e, h_prev, z, r, g, h, probs in reversed(cache):
        dlogits = probs.copy()
        dlogits[target_id] -= 1.0
        grads["W_o"] += np.outer(h, dlogits)
        grads["b_o"] += dlogits
        dh = dlogits @ weights["W_o"].T + dh_next

        dh_prev = dh * (1.0 - z)
        dz = dh * (g - h_prev)
        dg = dh * z
        dg_raw = dg * (1.0 - g * g)
        grads["W_h"] += np.outer(e, dg_raw)
        grads["U_h"] += np.outer(r * h_prev, dg_raw)
        grads["b_h"] += dg_raw

        drh = dg_raw @ weights["U_h"].T
        dr = drh * h_prev
        dh_prev += drh * r
        dr_raw = dr * r * (1.0 - r)
        grads["W_r"] += np.outer(e, dr_raw)
        grads["U_r"] += np.outer(h_prev, dr_raw)
        grads["b_r"] += dr_raw

        dz_raw = dz * z * (1.0 - z)
        grads["W_z"] += np.outer(e, dz_raw)
        grads["U_z"] += np.outer(h_prev, dz_raw)
        grads["b_z"] += dz_raw

        de = (
            dz_raw @ weights["W_z"].T
            + dr_raw @ weights["W_r"].T
            + dg_raw @ weights["W_h"].T
        )
        grads["E"][input_id] += de
        dh_prev += dz_raw @ weights["U_z"].T
        dh_prev += dr_raw @ weights["U_r"].T
        dh_next = dh_prev

    if token_count:
        for name in grads:
            grads[name] /= token_count
    return total_loss, grads, token_count


def _split_likelihood(
    records: Sequence[dict[str, object]],
    splits: set[str],
    weights: Mapping[str, np.ndarray],
    vocabulary: Sequence[str],
) -> dict[str, object]:
    build = build_training_examples(records, allowed_splits=splits)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    token_count = 0
    nll = 0.0
    for example in build.examples:
        ids = _stream_to_ids(list(example["tokens"]), token_to_id)
        if len(ids) < 2:
            continue
        loss, count = _sequence_loss(weights, ids)
        nll += loss
        token_count += count
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
        "dev_perplexity": math.exp(min(50.0, avg_nll)),
    }


def _sequence_loss(weights: Mapping[str, np.ndarray], ids: Sequence[int]) -> tuple[float, int]:
    h = np.zeros(weights["b_z"].shape[0])
    loss = 0.0
    count = 0
    for input_id, target_id in zip(ids[:-1], ids[1:]):
        h, logits = _step(weights, h, input_id)
        probs = _softmax(logits)
        loss -= math.log(max(float(probs[target_id]), 1e-12))
        count += 1
    return loss, count


def _generate_tokens(
    model: NeuralHaikuModel,
    *,
    prompt: str,
    observer: str | None,
    temperature: float,
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
    anchor = _prompt_anchor(prompt)
    tokens.extend(anchor)
    token_to_id = model.token_to_id
    h = np.zeros(model.hidden_size)
    logits = np.zeros(len(model.vocabulary))
    for token in tokens:
        h, logits = _step(model.weights, h, token_to_id.get(token, token_to_id[UNK_TOKEN]))

    line_index = 0
    line_chars = [len(anchor), 0, 0]
    for _ in range(sum(DEFAULT_MAX_LINE_CHARS) + 12):
        allowed = _allowed_tokens(model, line_index, line_chars[line_index])
        token = _sample_allowed(model.vocabulary, logits, allowed, temperature, rng)
        tokens.append(token)
        h, logits = _step(model.weights, h, token_to_id.get(token, token_to_id[UNK_TOKEN]))
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


def _allowed_tokens(model: NeuralHaikuModel, line_index: int, current_length: int) -> set[str]:
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


def _sample_allowed(
    vocabulary: Sequence[str],
    logits: np.ndarray,
    allowed: set[str],
    temperature: float,
    rng: random.Random,
) -> str:
    allowed_ids = [index for index, token in enumerate(vocabulary) if token in allowed]
    if not allowed_ids:
        return END_TOKEN
    scaled = np.array([logits[index] for index in allowed_ids], dtype=np.float64) / temperature
    probs = _softmax(scaled)
    threshold = rng.random()
    running = 0.0
    for index, probability in sorted(zip(allowed_ids, probs), key=lambda item: vocabulary[item[0]]):
        running += float(probability)
        if running >= threshold:
            return vocabulary[index]
    return vocabulary[allowed_ids[-1]]


def _step(weights: Mapping[str, np.ndarray], h: np.ndarray, token_id: int) -> tuple[np.ndarray, np.ndarray]:
    e = weights["E"][token_id]
    z = _sigmoid(e @ weights["W_z"] + h @ weights["U_z"] + weights["b_z"])
    r = _sigmoid(e @ weights["W_r"] + h @ weights["U_r"] + weights["b_r"])
    g = np.tanh(e @ weights["W_h"] + (r * h) @ weights["U_h"] + weights["b_h"])
    next_h = (1.0 - z) * h + z * g
    logits = next_h @ weights["W_o"] + weights["b_o"]
    return next_h, logits


def _init_weights(
    *,
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    scale = 0.08
    recurrent_scale = 1.0 / math.sqrt(hidden_size)
    return {
        "E": rng.normal(0.0, scale, (vocab_size, embedding_dim)),
        "W_z": rng.normal(0.0, scale, (embedding_dim, hidden_size)),
        "U_z": rng.normal(0.0, recurrent_scale, (hidden_size, hidden_size)),
        "b_z": np.zeros(hidden_size),
        "W_r": rng.normal(0.0, scale, (embedding_dim, hidden_size)),
        "U_r": rng.normal(0.0, recurrent_scale, (hidden_size, hidden_size)),
        "b_r": np.zeros(hidden_size),
        "W_h": rng.normal(0.0, scale, (embedding_dim, hidden_size)),
        "U_h": rng.normal(0.0, recurrent_scale, (hidden_size, hidden_size)),
        "b_h": np.zeros(hidden_size),
        "W_o": rng.normal(0.0, scale, (hidden_size, vocab_size)),
        "b_o": np.zeros(vocab_size),
    }


def _clip_grads(grads: Mapping[str, np.ndarray], *, max_norm: float) -> None:
    norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values()))
    if norm <= max_norm or norm == 0.0:
        return
    scale = max_norm / norm
    for grad in grads.values():
        grad *= scale


def _build_neural_vocabulary(streams: Sequence[Sequence[str]]) -> tuple[str, ...]:
    seen = {UNK_TOKEN}
    for stream in streams:
        seen.update(stream)
    return tuple(sorted(seen, key=_token_sort_key))


def _stream_to_ids(stream: Sequence[str], token_to_id: Mapping[str, int]) -> list[int]:
    unknown_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(token, unknown_id) for token in stream]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _observer_from_prompt(prompt: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", prompt.casefold()))
    if words & {"disk", "drive", "filesystem", "file", "bytes", "root"}:
        return "disk"
    if words & {"network", "localhost", "latency", "packet", "packets", "ping", "loopback"}:
        return "network"
    if words & {"process", "cpu", "load", "memory", "thread"}:
        return "process"
    return "unknown"


def _prompt_anchor(prompt: str) -> list[str]:
    words = _prompt_words(prompt)
    for keywords, anchor in PROMPT_TOPIC_ANCHORS:
        if words & keywords:
            return list(anchor)
    return []


def _prompt_rejection_reason(prompt: str, text: str) -> str | None:
    words = _prompt_words(prompt)
    if not words:
        return None
    poem_words = _prompt_words(text)
    if words & poem_words:
        return None
    for keywords, _anchor in PROMPT_TOPIC_ANCHORS:
        if words & keywords and poem_words & keywords:
            return None
    return "prompt_topic_overlap"


def _prompt_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_./:%']+", text.casefold()))


def _poem_hash(text: str) -> str:
    normalized = normalize_poem_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _metadata_token(key: str, value: str) -> str:
    return f"<{key}={quote(value, safe='._~-')}>"


def _is_metadata_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">") and "=" in token


def _token_sort_key(token: str) -> tuple[int, str]:
    if token == UNK_TOKEN:
        return (0, token)
    if token in CONTROL_TOKENS:
        return (1, token)
    if _is_metadata_token(token):
        return (2, token)
    return (3, token)


def _checkpoint_reference(checkpoint: Path, metadata_dir: Path) -> str:
    try:
        return os.path.relpath(checkpoint.resolve(), metadata_dir.resolve())
    except ValueError:
        return str(checkpoint)


if __name__ == "__main__":
    raise SystemExit(train_main())
