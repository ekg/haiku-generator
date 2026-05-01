# Local Haiku Operation

This is the minimal local operation path for generating a haiku from the
validated learned model artifact. It uses only the Python standard library and
repo-local files; it does not call cloud APIs and does not require privileged
installs.

## Command

Run from the repository root:

```bash
python scripts/generate_local_haiku.py \
  --prompt "Write a haiku about localhost latency." \
  --seed 301
```

Another deterministic example:

```bash
python scripts/generate_local_haiku.py \
  --prompt "Write a haiku about disk bytes in a build cache." \
  --seed 302
```

The basic user command is `--prompt` plus `--seed`. The command currently
defaults to the validated CPU n-gram model artifact:

```text
artifacts/local-haiku/ngram/baseline-validation/model.json.gz
```

Advanced runs can override the artifact with `--model`, but callers should keep
the same `--prompt` and `--seed` shape so a stronger future local model can be
selected behind the same command.

## Device Selection

The default artifact is a CPU-only character/control-token n-gram model. It does
not use GPU libraries. CPU is selected implicitly by the artifact and by the
standard-library runtime. GPU acceleration is not currently used; future stronger
local models should preserve the same prompt command and document any optional
device flag separately.

## Latency

Expected latency on a normal development laptop is interactive: usually well
under one second for the default retry budget. The command samples candidates
locally and validates each candidate before printing it.

## Failure And Retry Behavior

The command retries candidate generation up to `--max-attempts` times, default
`200`. A candidate must pass local validation before it is printed:

- exactly three nonempty lines
- no repeated full line
- no repeated three-word phrase
- per-line character limits
- no exact duplicate of a training poem
- prompt/topic overlap when a prompt is supplied

If no candidate passes validation, the command exits with status `2` and prints
a clear error to stderr, for example:

```text
ERROR: no accepted candidate after 0 attempts
```

Missing or unreadable model artifacts also exit with status `2`.

## Local Smoke Validation

The operation path was smoke-validated with three prompt/seed pairs and the
default validated artifact:

```bash
python scripts/generate_local_haiku.py --prompt "Write a haiku about localhost latency." --seed 301
python scripts/generate_local_haiku.py --prompt "Write a haiku about disk bytes in a build cache." --seed 302
python scripts/generate_local_haiku.py --prompt "Write a haiku about process memory at dawn." --seed 303
```

To capture samples for evaluation or later inspection:

```bash
python scripts/generate_local_haiku.py \
  --prompt "Write a haiku about localhost latency." \
  --seed 301 \
  --samples-out artifacts/local-haiku/ngram/local-operation-smoke/samples.jsonl
```
