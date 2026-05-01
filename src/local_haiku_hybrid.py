"""Hybrid local haiku decoder.

The n-gram model is treated as a compact graph of legal next-token edges. The
tiny neural model does not generate freely; it only reranks those edges.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Sequence

import numpy as np

import local_haiku_neural as neural
import local_haiku_ngram as ngram
from local_haiku_tokenizer import (
    END_TOKEN,
    HAIKU_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    PROMPT_END_TOKEN,
    PROMPT_TOKEN,
    decode_tokens,
)


DEFAULT_NEURAL_WEIGHT = 0.6
DEFAULT_TEMPERATURE = 0.9
DEFAULT_CANDIDATE_POOL = 4


def generate_haiku(
    ngram_model: ngram.NGramModel,
    neural_model: neural.NeuralHaikuModel,
    *,
    prompt: str = "",
    observer: str | None = None,
    seed: int | None = None,
    neural_weight: float = DEFAULT_NEURAL_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int | None = None,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    max_attempts: int = 200,
) -> tuple[str, dict[str, object]]:
    """Generate one accepted haiku by walking n-gram edges with neural reranking."""

    if neural_weight < 0:
        raise ValueError("neural_weight must be >= 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be >= 1")

    rng = random.Random(seed)
    accepted: list[tuple[int, int, str, list[str]]] = []
    for attempt in range(1, max_attempts + 1):
        tokens = _generate_tokens(
            ngram_model,
            neural_model,
            prompt=prompt,
            observer=observer,
            neural_weight=neural_weight,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        try:
            decoded = decode_tokens(tokens)
        except ValueError:
            continue
        text = "\n".join(line.strip() for line in decoded.lines)
        rejection = ngram.rejection_reason(ngram_model, text)
        if rejection is None:
            rejection = ngram._prompt_rejection_reason(prompt, text)
        if rejection is None:
            accepted.append((_candidate_quality_penalty(text), attempt, text, tokens))
            if len(accepted) >= candidate_pool:
                break

    if accepted:
        quality_penalty, attempt, text, tokens = min(accepted, key=lambda item: (item[0], item[1], item[2]))
        return text, {
            "attempt": attempt,
            "accepted_candidates": len(accepted),
            "candidate_quality_penalty": quality_penalty,
            "prompt": prompt,
            "observer": observer or ngram._observer_from_prompt(prompt),
            "seed": seed,
            "neural_weight": neural_weight,
            "temperature": temperature,
            "top_k": top_k,
            "candidate_pool": candidate_pool,
            "tokens": tokens,
            "decoder": "ngram-graph-neural-rerank",
        }

    raise RuntimeError(f"no accepted candidate after {max_attempts} attempts")


def generate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate with a hybrid local decoder: n-gram graph edges reranked by a tiny GRU."
    )
    parser.add_argument("--ngram-model", required=True, help="N-gram JSON gzip model artifact.")
    parser.add_argument("--neural-model", required=True, help="Neural model metadata JSON artifact.")
    parser.add_argument("--prompt", default="", help="Prompt text to condition the control-token prefix.")
    parser.add_argument("--observer", help="Observer tag override, for example disk/network/process.")
    parser.add_argument("--seed", type=int, help="Deterministic random seed.")
    parser.add_argument("--neural-weight", type=float, default=DEFAULT_NEURAL_WEIGHT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, help="Limit sampling to the highest scoring graph edges.")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help="Return the cleanest candidate from this many accepted samples.",
    )
    parser.add_argument("--max-attempts", type=int, default=200, help="Candidate retry budget.")
    parser.add_argument("--samples-out", help="Optional JSONL file for the accepted sample.")
    args = parser.parse_args(argv)

    try:
        ngram_model = ngram.load_model(args.ngram_model)
        neural_model = neural.load_model(args.neural_model)
        poem, metadata = generate_haiku(
            ngram_model,
            neural_model,
            prompt=args.prompt,
            observer=args.observer,
            seed=args.seed,
            neural_weight=args.neural_weight,
            temperature=args.temperature,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            max_attempts=args.max_attempts,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: local haiku model artifact not found: {exc.filename}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: failed to load or use local haiku model artifacts: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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


def _generate_tokens(
    ngram_model: ngram.NGramModel,
    neural_model: neural.NeuralHaikuModel,
    *,
    prompt: str,
    observer: str | None,
    neural_weight: float,
    temperature: float,
    top_k: int | None,
    rng: random.Random,
) -> list[str]:
    observer_value = observer or ngram._observer_from_prompt(prompt)
    tokens = [
        HAIKU_TOKEN,
        "<LANG=en>",
        ngram._metadata_token("OBSERVER", observer_value),
        "<SOURCE=repo-local>",
        PROMPT_TOKEN,
        *list(prompt),
        PROMPT_END_TOKEN,
        L1_TOKEN,
    ]
    anchor = ngram._prompt_anchor(prompt)
    tokens.extend(anchor)

    token_to_id = neural_model.token_to_id
    unknown_id = token_to_id.get(neural.UNK_TOKEN)
    h = np.zeros(neural_model.hidden_size)
    logits = np.zeros(len(neural_model.vocabulary))
    for token in tokens:
        token_id = token_to_id.get(token, unknown_id)
        if token_id is not None:
            h, logits = neural._step(neural_model.weights, h, token_id)

    line_index = 0
    line_chars = [len(anchor), 0, 0]
    for _ in range(sum(ngram.DEFAULT_MAX_LINE_CHARS) + 12):
        allowed = ngram._allowed_tokens(ngram_model, tokens, line_index, line_chars[line_index])
        graph_weights = _graph_edge_weights(ngram_model, tokens, allowed)
        token = _sample_hybrid_edge(
            neural_model,
            logits,
            graph_weights,
            neural_weight=neural_weight,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        tokens.append(token)

        token_id = token_to_id.get(token, unknown_id)
        if token_id is not None:
            h, logits = neural._step(neural_model.weights, h, token_id)

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


def _graph_edge_weights(
    model: ngram.NGramModel,
    tokens: Sequence[str],
    allowed: set[str],
) -> Counter[str]:
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
            return weights

    unigram = model.counts.get(1, {}).get((), Counter())
    for token in allowed:
        if unigram.get(token, 0):
            weights[token] += unigram[token]
    if weights:
        return weights
    return Counter({token: 1 for token in allowed})


def _sample_hybrid_edge(
    neural_model: neural.NeuralHaikuModel,
    logits: np.ndarray,
    graph_weights: Counter[str],
    *,
    neural_weight: float,
    temperature: float,
    top_k: int | None,
    rng: random.Random,
) -> str:
    if not graph_weights:
        return END_TOKEN

    vocabulary = neural_model.vocabulary
    token_to_id = neural_model.token_to_id
    unknown_id = token_to_id.get(neural.UNK_TOKEN)
    total_graph_weight = float(sum(graph_weights.values()))
    scored: list[tuple[str, float]] = []
    for token, weight in graph_weights.items():
        token_id = token_to_id.get(token, unknown_id)
        neural_logit = float(logits[token_id]) if token_id is not None else 0.0
        graph_log_probability = math.log(max(float(weight) / total_graph_weight, 1e-12))
        scored.append((token, graph_log_probability + neural_weight * neural_logit))

    scored.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        scored = scored[:top_k]

    values = np.array([score for _token, score in scored], dtype=np.float64) / temperature
    probabilities = neural._softmax(values)
    threshold = rng.random()
    running = 0.0
    for (token, _score), probability in zip(scored, probabilities):
        running += float(probability)
        if running >= threshold:
            return token
    return scored[-1][0]


def _candidate_quality_penalty(text: str) -> int:
    """Prefer accepted candidates without obvious fragment and repetition artifacts."""

    word_lines = [re.findall(r"[A-Za-z0-9_./:%'-]+", line.casefold()) for line in text.splitlines()]
    words = [word for line in word_lines for word in line]
    counts = Counter(words)
    short_words = [word for word in words if len(word) <= 2 and not word.isdigit()]
    numeric_words = [word for word in words if any(ch.isdigit() for ch in word)]
    line_end_fragments = [
        line[-1]
        for line in word_lines
        if line and (len(line[-1]) <= 2 or line[-1].endswith("-")) and not line[-1].isdigit()
    ]
    repeated_words = [word for word, count in counts.items() if count > 1]
    return len(short_words) + len(numeric_words) + (2 * len(line_end_fragments)) + len(repeated_words)


if __name__ == "__main__":
    raise SystemExit(generate_main())
