# OpenRouter Model Research for Tiered Agent System

**Date:** 2026-03-23
**Source:** OpenRouter API (`/api/v1/models`) + web research
**Total models available:** 349

## Summary: Recommended Tier Configuration

| Tier | Role | Model ID | Context | Input $/M tok | Output $/M tok | Notes |
|------|------|----------|---------|---------------|----------------|-------|
| **Fast/Cheap** | Triage, eval, summarization | `deepseek/deepseek-v3.2` | 163K | $0.26 | $0.38 | Best price/performance ratio. GPT-5 class at 1/50th cost |
| **Fast/Cheap (free)** | Budget fallback | `nvidia/nemotron-3-super-120b-a12b:free` | 262K | $0 | $0 | Hybrid Mamba-Transformer, 120B/12B active. Free tier rate limits: 20 req/min, 200/day |
| **Standard** | Main task agents, code gen | `minimax/minimax-m2.5` | 196K | $0.20 | $1.17 | #1 most-used model on OpenRouter. 230B MoE, 10B active. SWE-Bench 80.2% |
| **Standard (alt)** | Code-focused tasks | `qwen/qwen3-coder` | 262K | $0.22 | $1.00 | 480B/35B active MoE. Best free coding model also available as `qwen/qwen3-coder:free` |
| **Premium** | Hard problems, coordination | `minimax/minimax-m2.7` | 204K | $0.30 | $1.20 | Self-evolving model. SWE-Pro 56.2%, 97% skill adherence, multi-agent native |
| **Premium (alt)** | Reasoning-heavy tasks | `qwen/qwen3.5-397b-a17b` | 262K | $0.39 | $2.34 | Hybrid Gated DeltaNet + MoE architecture. SOTA across reasoning, code, vision |
| **Long-Context** | Large context tasks (128K+) | `qwen/qwen3.5-flash-02-23` | 1M | $0.065 | $0.26 | 1M context at dirt-cheap prices |
| **Long-Context (alt)** | Budget long context | `google/gemini-2.0-flash-001` | 1M | $0.10 | $0.40 | Google's proven 1M context, very reliable |
| **Long-Context (premium)** | Large context + quality | `qwen/qwen3-coder-flash` | 1M | $0.195 | $0.975 | 1M context with strong code capabilities |

## Tier Details

### Fast/Cheap Tier (Haiku Equivalent)

**Primary: `deepseek/deepseek-v3.2`**
- 163,840 token context, $0.26/$0.38 per M tokens
- DeepSeek Sparse Attention (DSA) for efficient long-context
- ~90% of GPT-5.4 performance at a fraction of the cost
- Variants: `deepseek/deepseek-v3.2-exp` ($0.27/$0.41), `deepseek/deepseek-v3.2-speciale` ($0.40/$1.20 — higher quality)

**Alternatives:**
| Model ID | Context | Input $/M | Output $/M | Notes |
|----------|---------|-----------|------------|-------|
| `qwen/qwen-turbo` | 131K | $0.0325 | $0.13 | Extremely cheap, good for simple tasks |
| `qwen/qwen3.5-9b` | 256K | $0.05 | $0.15 | Small but capable, long context |
| `bytedance-seed/seed-2.0-mini` | 262K | $0.10 | $0.40 | Good price, long context |
| `mistralai/mistral-small-3.2-24b-instruct` | 128K | $0.075 | $0.20 | Solid small model from Mistral |
| `nvidia/nemotron-nano-9b-v2` | 131K | $0.04 | $0.16 | Tiny, fast (also free variant) |

**Free options for this tier:**
| Model ID | Context | Notes |
|----------|---------|-------|
| `nvidia/nemotron-3-super-120b-a12b:free` | 262K | Best free model: hybrid Mamba-Transformer, 120B total/12B active |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 256K | Smaller NVIDIA hybrid, good for light tasks |
| `nvidia/nemotron-nano-9b-v2:free` | 128K | Tiny, fast |
| `qwen/qwen3-4b:free` | 40K | Smallest Qwen, very fast |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 128K | Solid Mistral small |
| `openai/gpt-oss-120b:free` | 131K | OpenAI's first open-weight model (Apache 2.0) |
| `minimax/minimax-m2.5:free` | 196K | Same great model, rate-limited |

### Standard Tier (Sonnet Equivalent)

**Primary: `minimax/minimax-m2.5`**
- 196,608 token context, max 65,536 output tokens
- 230B MoE with only 10B active parameters → fast inference
- SWE-Bench Verified: 80.2%, Multi-SWE-Bench: 51.3%, BrowseComp: 76.3%
- #1 most-used model on OpenRouter by token volume (2.45T+ tokens/week as of Feb 2026)
- Strong at office productivity (Word/Excel/PowerPoint), agent workflows, code
- $0.20/$1.17 per M tokens — excellent value for the quality

**Code-focused alternative: `qwen/qwen3-coder`**
- 480B/35B active MoE, 262K context
- $0.22/$1.00 per M tokens
- State-of-the-art code generation
- Free variant available: `qwen/qwen3-coder:free` (262K context, rate-limited)

**Other alternatives:**
| Model ID | Context | Input $/M | Output $/M | Notes |
|----------|---------|-----------|------------|-------|
| `qwen/qwen3-coder-next` | 262K | $0.12 | $0.75 | Newer coder variant, slightly cheaper |
| `moonshotai/kimi-k2.5` | 262K | $0.45 | $2.20 | Strong reasoning |
| `deepseek/deepseek-v3.2-speciale` | 163K | $0.40 | $1.20 | Enhanced V3.2 |
| `mistralai/devstral-2512` | 262K | $0.40 | $2.00 | Mistral's dev-focused model |
| `qwen/qwen3-235b-a22b` | 131K | $0.455 | $1.82 | Large MoE, strong general |
| `z-ai/glm-4.7` | 202K | $0.39 | $1.75 | Zhipu's latest |

### Premium Tier (Opus Equivalent)

**Primary: `minimax/minimax-m2.7`**
- 204,800 token context
- $0.30/$1.20 per M tokens — remarkably cheap for premium tier
- Self-evolving: handles 30-50% of its own RL development workflow
- SWE-Pro: 56.2%, TerminalBench 2: 57.0%, GDPval-AA: 1495 ELO
- 97% skill adherence with 40+ complex skills (>2000 tokens each)
- Multi-agent native: built for agent teams, complex skills, dynamic tool search
- Ideal for orchestration, verification, and hard problem-solving

**Reasoning alternative: `qwen/qwen3.5-397b-a17b`**
- 397B total / 17B active parameters
- Hybrid architecture: Gated DeltaNet linear attention + sparse MoE
- 262K context, $0.39/$2.34 per M tokens
- SOTA across language understanding, reasoning, code, agent tasks, vision
- Native vision-language model (can process images)

**Other alternatives:**
| Model ID | Context | Input $/M | Output $/M | Notes |
|----------|---------|-----------|------------|-------|
| `qwen/qwen3-max` | 262K | $0.78 | $3.90 | Qwen's most capable, with thinking variant |
| `x-ai/grok-4` | 256K | $3.00 | $15.00 | xAI's flagship, very expensive |
| `google/gemini-3.1-pro-preview` | 1M | $2.00 | $12.00 | Google's latest pro, 1M context |
| `xiaomi/mimo-v2-pro` | 1M | $1.00 | $3.00 | Strong reasoner with 1M context |

### Long-Context Tier (128K+)

For tasks requiring massive context windows:

| Model ID | Context | Input $/M | Output $/M | Notes |
|----------|---------|-----------|------------|-------|
| `qwen/qwen3.5-flash-02-23` | **1M** | $0.065 | $0.26 | Cheapest 1M context option |
| `google/gemini-2.0-flash-001` | **1M** | $0.10 | $0.40 | Proven reliability, Google infrastructure |
| `qwen/qwen3-coder-flash` | **1M** | $0.195 | $0.975 | Code-focused with 1M context |
| `qwen/qwen-plus` | **1M** | $0.26 | $0.78 | General-purpose Qwen with 1M |
| `qwen/qwen3-coder-plus` | **1M** | $0.65 | $3.25 | Premium code + 1M context |
| `amazon/nova-2-lite-v1` | **1M** | $0.30 | $2.50 | Amazon's 1M context model |
| `google/gemini-3.1-flash-lite-preview` | **1M** | $0.25 | $1.50 | Google's latest lite with 1M |
| `meta-llama/llama-4-maverick` | **1M** | $0.15 | $0.60 | Meta's long-context MoE |
| `x-ai/grok-4-fast` | **2M** | $0.20 | $0.50 | Largest context available (2M!) |

## Specific Models the User Asked About

### MiniMax M2.5 / M2.7
- **M2.5** (`minimax/minimax-m2.5`): Recommended as **Standard tier**. 230B MoE/10B active, 196K ctx. $0.20/$1.17. SWE-Bench 80.2%. Most popular model on OpenRouter by volume. Also available free.
- **M2.7** (`minimax/minimax-m2.7`): Recommended as **Premium tier**. 204K ctx. $0.30/$1.20. Self-evolving, multi-agent native. SWE-Pro 56.2%.
- **M2.5-Lightning**: Doubles speed to 100 tok/s at ~$0.30/$2.40 (not separately listed in API but may be available as a variant).

### NVIDIA Gated DeltaNet Models
- No pure NVIDIA Gated DeltaNet model is directly available on OpenRouter under NVIDIA's name.
- **Qwen3.5-397B-A17B** (`qwen/qwen3.5-397b-a17b`) uses the Gated DeltaNet architecture (64 linear attention heads for V, 16 for QK). This is the primary way to access Gated DeltaNet on OpenRouter.
- **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) uses a hybrid Mamba-Transformer MoE architecture (related but distinct from pure Gated DeltaNet). Available free!
- **Nemotron-H family** (8B, 47B, 56B) with hybrid Mamba2-Transformer: not directly listed on OpenRouter yet, but the Nemotron 3 line incorporates similar ideas.

### Mamba2 Architecture Models
- **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`): Hybrid Mamba-Transformer MoE, 120B/12B active, 262K context. Free and paid variants. Multi-token prediction for 50%+ faster generation.
- **NVIDIA Nemotron 3 Nano** (`nvidia/nemotron-3-nano-30b-a3b`): Smaller hybrid, 256K context. Free and paid.
- **Codestral Mamba** (`mistralai/codestral-mamba`): Pure Mamba2-based code model from Mistral, 7.3B params, 256K context. Optimized for code and retrieval. Note: may have limited availability/deprecated status.
- **AI21 Jamba Large 1.7** (`ai21/jamba-large-1.7`): Hybrid Mamba model, 256K context, $2.00/$8.00. Expensive but proven.

## Models to Avoid

| Model | Reason |
|-------|--------|
| `openai/gpt-4` / `gpt-4-turbo` | Legacy, expensive ($30/$60 per M), outclassed by newer models |
| `openai/o1-pro` | $150/$600 per M tokens — extremely expensive, niche use only |
| `openai/gpt-5.4-pro` | $30/$180 per M tokens — very expensive for marginal gains |
| `openai/gpt-5.2-pro` | $21/$168 per M tokens |
| `alpindale/goliath-120b` | 6K context, old, $3.75/$7.50 |
| `inflection/inflection-3-*` | Only 8K context, $2.50/$10.00 — bad value |
| `mistralai/mixtral-8x22b-instruct` | $2.00/$6.00 — outclassed by newer Mistral models |
| RP/creative models (sao10k, thedrummer, etc.) | Not designed for agent/code tasks |
| Any model with <8K context | Too small for agent workflows |

## Reliability Notes

- **OpenRouter platform**: Had outages on Feb 17 and Feb 19, 2026 (38 and 35 min respectively). Root cause: third-party caching dependency + DDoS. They now return 503 instead of 401 for infrastructure issues. Use model fallbacks (`X-Fallback-Models` header) for resilience.
- **Free models**: Rate-limited to 20 req/min, 200 req/day. Not suitable as sole tier for production workloads but good for cost optimization.
- **DeepSeek models**: Historically had intermittent availability due to high demand. Use OpenRouter's multi-provider routing for better reliability.
- **MiniMax models**: Proven at scale (2.45T tokens/week). Reliable.
- **Google/OpenAI models**: Most reliable infrastructure backing.
- **Recommendation**: Always configure `X-Fallback-Models` header with alternatives per tier.

## Recommended Workgraph Configuration

```
# Tier configuration for workgraph agent system
# Format: role -> primary model (fallback)

[fast]
# Triage, evaluation, summarization, quick tasks
primary = "deepseek/deepseek-v3.2"
fallback = "nvidia/nemotron-3-super-120b-a12b:free"
budget_fallback = "qwen/qwen-turbo"

[standard]
# Main task agents, code generation, reasoning
primary = "minimax/minimax-m2.5"
code_focused = "qwen/qwen3-coder"
fallback = "qwen/qwen3-coder-next"
free_fallback = "qwen/qwen3-coder:free"

[premium]
# Hard problems, coordination, verification, orchestration
primary = "minimax/minimax-m2.7"
reasoning = "qwen/qwen3.5-397b-a17b"
fallback = "qwen/qwen3-max"

[long_context]
# Tasks needing >128K context
primary = "qwen/qwen3.5-flash-02-23"        # 1M ctx, cheapest
code = "qwen/qwen3-coder-flash"             # 1M ctx, code-focused
premium = "google/gemini-3.1-pro-preview"    # 1M ctx, highest quality
budget = "google/gemini-2.0-flash-001"       # 1M ctx, proven reliable
extreme = "x-ai/grok-4-fast"                 # 2M ctx!
```

## Cost Comparison (per 1M tokens, input/output)

| Category | Model | Input | Output | Total (50/50 mix) |
|----------|-------|-------|--------|-------------------|
| **Ultra-cheap** | `qwen/qwen-turbo` | $0.033 | $0.13 | $0.08 |
| **Free** | `nvidia/nemotron-3-super-120b-a12b:free` | $0 | $0 | $0 |
| **Fast** | `deepseek/deepseek-v3.2` | $0.26 | $0.38 | $0.32 |
| **Standard** | `minimax/minimax-m2.5` | $0.20 | $1.17 | $0.69 |
| **Standard (code)** | `qwen/qwen3-coder` | $0.22 | $1.00 | $0.61 |
| **Premium** | `minimax/minimax-m2.7` | $0.30 | $1.20 | $0.75 |
| **Premium (reasoning)** | `qwen/qwen3.5-397b-a17b` | $0.39 | $2.34 | $1.37 |
| **Long-ctx cheap** | `qwen/qwen3.5-flash-02-23` | $0.065 | $0.26 | $0.16 |
| *For reference: Claude Sonnet 4.6* | `anthropic/claude-sonnet-4.6` | $3.00 | $15.00 | $9.00 |
| *For reference: Claude Opus 4.6* | `anthropic/claude-opus-4.6` | $5.00 | $25.00 | $15.00 |

The recommended tiers are **10-100x cheaper** than equivalent Anthropic models while delivering competitive quality for agent workflows.

## Architecture Innovation Notes

The most interesting architectural trend on OpenRouter in 2026:
1. **Mixture-of-Experts (MoE)** dominates: MiniMax M2.5 (230B/10B), Qwen3.5-397B (397B/17B), Nemotron 3 Super (120B/12B)
2. **Hybrid attention**: Gated DeltaNet (Qwen3.5), Mamba-Transformer hybrids (Nemotron 3), SSM+Attention (Jamba)
3. **Linear attention models**: Enable theoretically infinite context with linear time complexity
4. **Multi-token prediction**: Nemotron 3 Super generates 50%+ more tokens per forward pass
5. **Self-evolution**: MiniMax M2.7 handles 30-50% of its own RL training workflow
