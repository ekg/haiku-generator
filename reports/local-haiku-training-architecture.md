# Local Haiku Training Architecture Recommendation

Date: 2026-05-01

## Recommendation

Build the first local learned generator as a CPU-only character or token n-gram model with explicit line-boundary and prompt/context tokens. This is the smallest model that genuinely learns from the existing corpus rather than only templating: it estimates transition probabilities from examples, can condition on observer/topic tokens, and can be trained in seconds on the current repo-local haiku set.

Use the following staged path:

1. Trivial sanity model: retrieval plus line-preserving template.
2. First learned model: smoothed character or compact-token n-gram with constrained three-line decoding.
3. Stronger learned model: tiny GRU/LSTM language model trained on CPU.
4. Optional GPU path: PyTorch ROCm probe for AMD laptops; use GPU only when the probe is clean.

The CPU path is the required baseline. AMD GPU support should be treated as optional acceleration, not a dependency for correctness.

## Repo And Data Starting Point

The repo currently has about 2,519 local `.haiku` files under `haikus/`. Many include YAML frontmatter and three poem lines, so the ingestion step should strip frontmatter, preserve observer/source metadata, normalize line breaks, deduplicate by normalized poem text, and split train/validation/test by file hash or timestamp.

Minimum useful data:

- Sanity/retrieval: 20-50 examples.
- N-gram: 200+ examples; current repo-local corpus is enough for a first smoke model.
- GRU/LSTM: 1,000-10,000 examples; current corpus is enough to overfit and generate style, but more diverse public-domain/permissive haiku will be needed for quality.
- Tiny Transformer: 10,000+ examples preferred; not the first success target.
- LoRA/fine-tune: thousands of prompt/haiku pairs and more setup; not needed for first local success.

## Baseline Options

| Option | Local feasibility | Training | Useful role | Main failure modes |
| --- | --- | --- | --- | --- |
| Retrieval/template | Excellent on CPU | None | Sanity check for data parsing, generation CLI, and validation harness | Does not learn; repeats corpus; weak prompt relevance |
| Markov/finite-state | Excellent on CPU | Seconds | Validates constrained decoding and line structure | Often incoherent; sparse transitions; syllable control is external |
| Smoothed n-gram | Excellent on CPU | Seconds to minutes | Recommended first learned model | Repetitive phrases; brittle prompt conditioning; needs backoff |
| Small GRU/LSTM | Good on CPU | Minutes to under a few hours depending size | Recommended stronger learned model | Overfits small corpus; slower sampling; needs PyTorch dependency |
| Tiny Transformer | Possible on CPU, better with GPU | Tens of minutes to hours | Later experiment if corpus grows | More moving parts; needs careful regularization |
| LoRA of small local LM | Possible but not first | GPU strongly preferred; CPU painful | Later quality path | Dependency weight, licensing/model choice, hard AMD setup |
| Retrieval-augmented template | Excellent on CPU | None or seconds for index | Useful fallback and prompt grounding | Still mostly recombination, not learned generation |

## Stage 0: Trivial Sanity Model

Implement a deterministic retrieval/template generator:

- Load normalized haikus.
- Select examples by observer/topic tags or lexical overlap with the prompt.
- Emit a three-line haiku unchanged, or fill a small template using repo-local nouns/adjectives.

Target:

- Model size: no model or a small JSON index under 1 MB.
- Training/build time: under 5 seconds.
- Dependencies: Python standard library.
- Validation: proves corpus parsing, CLI shape, output format, and duplicate filtering.

This is not the learned baseline. It should be used to keep the end-to-end pipeline runnable while learned models are being added.

## Stage 1: First Learned Model

Use a smoothed n-gram model over characters or compact tokens. Given the tokenization research is still upstream of implementation, choose this shape:

- Encode frontmatter-derived context as control tokens, for example `<observer=disk>` or `<prompt>network</prompt>`.
- Encode poem structure as `<HAIKU>`, `<L1>`, `<L2>`, `<L3>`, `<END>`.
- Train 3- to 5-gram counts with add-k/backoff smoothing.
- Decode with hard constraints: exactly three poem lines, maximum line character/token bounds, no repeated line, avoid exact full-poem memorization by rejecting corpus hashes.

Why this is the first learned model:

- It learns transition probabilities from corpus examples.
- It can train immediately on CPU.
- It has inspectable failure modes and no GPU dependency.
- It provides a baseline perplexity or held-out negative log-likelihood for later models.

Target:

- Model size: 1-20 MB as compressed JSON/msgpack/sqlite depending n and tokenization.
- Training time: seconds for 2,519 examples; likely under 1 minute on a normal laptop.
- Inference time: interactive on CPU.
- Data need: 200+ examples for smoke quality; 1,000+ better; current corpus is adequate for first success.
- Dependencies: Python standard library initially; optional `sqlite3`/`gzip` from stdlib for compact storage.

Failure modes:

- Character n-grams produce plausible texture but weak semantics.
- Word/token n-grams are more readable but sparse on a small corpus.
- Prompt conditioning may be shallow unless the corpus has reliable source/topic metadata.
- 5-7-5 syllable quality still needs a validator or soft rejection sampler.

## Stage 2: Stronger Learned CPU Model

Use a tiny GRU or LSTM character language model after the n-gram baseline is working.

Suggested configuration:

- Character vocabulary: roughly 80-200 symbols after normalization plus structure/control tokens.
- Embedding: 64-128.
- GRU/LSTM hidden size: 128-256.
- Layers: 1-2.
- Parameters: roughly 0.2M-1.5M.
- Sequence length: 128-256 characters.
- Batch size: 32-128 on CPU, tuned by memory.

Target:

- Dependencies: Python, PyTorch CPU wheel, optionally NumPy.
- Training time: 5-30 minutes for smoke runs; 30-120 minutes for fuller CPU experiments depending laptop.
- Checkpoint size: 1-10 MB.
- Data need: current corpus is enough to demonstrate learning and overfit; quality likely improves with 5,000-50,000 licensed examples.

Use this model when the n-gram pipeline has dataset splits, sample generation, and validation gates. It should reuse the same token stream and constraints where possible.

Failure modes:

- Overfitting and memorization with 2,519 similar repo-local poems.
- Syllable count and prompt relevance remain unreliable unless constrained or reranked.
- CPU training is acceptable for tiny models but iteration becomes slow as model size grows.

## Stage 3: Tiny Transformer Or LoRA Later

Tiny Transformer:

- Model: 2-4 layers, 2-4 attention heads, 128-256 hidden size.
- Parameters: roughly 1M-10M.
- CPU training: possible but slower, from tens of minutes to several hours.
- GPU value: meaningful for iteration speed once ROCm works.

LoRA/fine-tune of a small local model:

- Defer until the local dataset, prompt format, and validation harness are stable.
- Prefer inference through `llama.cpp` or similar for deployment, but do not depend on it for first training.
- CPU LoRA is usually not a good first-success path on a regular laptop; it adds dependency and model-license complexity before the project has a measured baseline.

## AMD GPU Feasibility

As of current AMD/PyTorch documentation, ROCm support is real but hardware and OS specific. Official paths include PyTorch ROCm wheels or AMD ROCm Docker images, and AMD also documents ONNX Runtime ROCm/MIGraphX providers. However, many laptop AMD GPUs and APUs are less predictable than workstation/server GPUs.

Probe steps for the hardware task:

1. Record OS, kernel, Python version, CPU, RAM, and disk.
2. Identify GPU/APU: `lspci -nn | grep -Ei 'vga|display|3d|amd|ati'`, `rocminfo` if installed, and `/opt/rocm/bin/rocminfo` if present.
3. Record ROCm/HIP visibility: `hipcc --version`, `rocm-smi`, `/dev/kfd`, `/dev/dri`, and user groups `video`/`render`.
4. Probe PyTorch without installing system packages:
   - `python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`
   - PyTorch uses the `torch.cuda` API name for ROCm availability.
5. Probe ONNX Runtime providers if installed:
   - `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
6. Classify result as `cpu`, `rocm-pytorch`, `onnx-rocm`, `llama-vulkan`, or `unknown`.

Fallback policy:

- If any ROCm probe fails, continue with CPU n-gram/GRU.
- Do not require privileged package installation in the training command.
- Keep GPU dependencies optional extras, such as `requirements-rocm.txt`, separate from CPU requirements.
- Prefer Docker for reproducible ROCm experiments when the host supports `/dev/kfd` and `/dev/dri`.

## Practical Runtime Paths

CPU-only path:

- Python stdlib for retrieval and n-gram.
- PyTorch CPU only for GRU/LSTM.
- Always available on normal laptop hardware.
- This should be the default in CI, smoke tests, and first implementation tasks.

ROCm/PyTorch path:

- Best for GRU/LSTM and tiny Transformer training if the AMD GPU is supported.
- Use official PyTorch/ROCm or AMD ROCm installation selectors for exact wheel/version.
- Keep versions pinned in an optional environment file after hardware probe confirms support.

ONNX path:

- Treat as inference acceleration, not first training infrastructure.
- Useful later if a PyTorch model is exported and the AMD ONNX Runtime providers are visible.

`llama.cpp` path:

- Useful later for local inference with small quantized LMs, especially via CPU or Vulkan/HIP backends.
- Not a practical first training path for the repo-local learned haiku baseline.

## Dependency Implications

Initial CPU baseline:

- `python>=3.10`
- No required ML dependency.
- Optional: `pytest` for tests.

First neural model:

- `torch` CPU wheel in a separate requirements file or optional extra.
- Keep checkpoint format simple: `torch.save` plus JSON metadata.

GPU optional:

- Separate probe command.
- Separate ROCm notes because exact install command depends on OS, ROCm version, Python version, and GPU architecture.
- Do not make ROCm a transitive dependency of the normal package or tests.

## Recommended First Success Target

First success should be:

- Train a 4-character or 3-token backoff n-gram model on normalized repo-local haikus.
- Save a model artifact under a generated artifacts directory.
- Generate a non-duplicate three-line haiku from a prompt or observer tag.
- Report held-out negative log-likelihood/perplexity and simple structural validation.

Target envelope:

- Training time: under 60 seconds on CPU.
- Model size: under 20 MB.
- Corpus: current 2,519 repo-local haikus enough for a demo.
- Quality bar: line structure, learned local diction, no exact memorized poem, some prompt/control-token influence.

## Graph Growth Notes

Existing downstream tasks already cover:

- `local-haiku-impl-first-model`: first learned local model.
- `local-haiku-impl-hardware-probe`: AMD GPU and environment probe.

I added a more specific CPU n-gram implementation task so the synthesis step has an unambiguous first learned baseline to route toward. The existing hardware probe task should remain the AMD GPU path.

## Sources Checked

- PyTorch local install selector: https://pytorch.org/get-started/locally/
- AMD ROCm Radeon/Ryzen compatibility: https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html
- ROCm PyTorch compatibility: https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html
- AMD ROCm PyTorch installation guidance: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html
- ONNX Runtime ROCm Execution Provider: https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
- AMD ONNX Runtime install notes: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html
- ROCm llama.cpp compatibility: https://rocm.docs.amd.com/en/develop/compatibility/ml-compatibility/llama-cpp-compatibility.html
