# Local Haiku Prompt Splits

These files define the stable prompt splits for local haiku model training and
evaluation. They are prompt-only splits; poem corpus splits are separate and
should be coordinated with the local corpus dataset task.

## Assignment Rule

Each prompt is assigned deterministically from the prompt text:

1. Normalize by trimming leading and trailing whitespace, lowercasing, and
   collapsing internal whitespace runs to a single ASCII space.
2. Compute `sha256(normalized_prompt)`.
3. Interpret the first eight hexadecimal digest characters as an unsigned
   integer and take `bucket = value % 100`.
4. Assign buckets `0..69` to `train.txt`, `70..84` to `dev.txt`, and `85..99`
   to `heldout.txt`.

Within each split, prompts are sorted by their normalized text for stable diffs.
The current prompt pool is intentionally small and stratified so disk, process,
network, local machine, nature, and news-like prompts appear in every split
where possible.

## Split Use

- `train.txt`: prompts allowed for training-time conditioning, prompt
  augmentation, tuning, and manual iteration.
- `dev.txt`: stable prompts visible during development and used on every local
  development evaluation run.
- `heldout.txt`: release and stage-gate prompts only. Do not use these prompts
  to tune prompt parsing, reranking thresholds, templates, model parameters, or
  other development decisions.
