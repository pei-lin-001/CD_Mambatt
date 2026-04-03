# Supervised Reproduction Alignment

This note records how the current supervised baseline maps the paper
`Mamba-attention: A self-supervised framework for efficient remaining useful life prediction`
into runnable code.

## Directly aligned from the paper

- Dataset: NASA C-MAPSS
- Input features: all `21` sensors
- Window size: `20`
- Sliding stride: `1`
- RUL cap: `125`
- Normalization:
  - `FD001` / `FD003`: global Z-score on the training split
  - `FD002` / `FD004`: condition-wise Z-score on the training split
- Mamba hyperparameters:
  - `d_state=16`
  - `d_conv=8`
  - `expand=2`
- Transformer head hyperparameters:
  - `num_transformer_layers=3`
  - `num_heads=7`
- Optimizer: Adam
- Learning rate: `1e-3`
- Dropout rate: `0.5`
- Validation split: engine-level `80/20`
- Model selection: choose the run with the lowest validation RMSE, then report test metrics

## Architecture implementation choices

The paper provides pseudocode in Algorithm 1 and text equations that are not perfectly
consistent, so the implementation follows the more explicit algorithm block.

- The Mamba encoder keeps the shape `(B, W, S)`.
- The encoder output is transposed to `(W, B, S)` before entering the Transformer head,
  matching Algorithm 1.
- Positional encoding is added after the transpose.
- Each Transformer block uses:
  - `LayerNorm -> MultiHeadAttention -> residual`
  - `LayerNorm -> FeedForward(ReLU) -> residual`
- Dropout is applied only after extracting the last time step and before the final dense
  layer, because this is the only dropout shown in Algorithm 1.

## Training procedure choices

- The supervised runner uses one fixed engine-level `80/20` split per subset and reuses it across
  all random-seed trials by default.
- A paper-closer alternative is now available through
  `--resample-split-per-seed`, which regenerates the engine-level `80/20` split for each run
  using that run's seed.
- The supervised runner defaults to validation on the last window of each validation engine.
  This matches the paper's final per-engine prediction protocol more closely than validating
  on every sliding window.
- The paper states that C-MAPSS experiments use multiple random seeds and mentions `50`
  iterations in Table 2. The code interprets this as:
  - `epochs=50` as the default per-run training budget
  - `num_runs=50` as the default number of random-seed trials
- If exact seed values are needed later, they can be overridden explicitly with `--seeds`.

## Remaining ambiguity

- The paper is internally inconsistent about Transformer normalization order:
  - `Algorithm 1` reads as `Pre-LN`
  - `Figure 1` visually resembles `Post-LN`
- The paper does not explicitly state whether the validation set is scored on only the last window
  per engine or on all validation windows.
- The paper does not explicitly publish the feed-forward hidden size `d_ff` of the Transformer
  block. The implementation uses `2048`, which matches the common default in standard
  Transformer implementations.
- The paper does not publish weight decay, a learning-rate schedule, or any gradient clipping
  policy.
- The paper does not publish the exact `expand` factor rationale beyond using the Mamba block.
- The paper's internal Mamba dimension `E` should not be confused with the external
  sequence width entering the Transformer. The text later states that the encoder maps
  `R^(B×W×S) -> R^(B×W×S)`, so keeping the external width at `21` is the more paper-faithful
  interpretation.

## Strongest current empirical variant

The most paper-faithful configuration is still the Table 2 setup. However, the strongest
current local `FD001` result is coming from a slightly different line:

- `transformer_norm_mode=post`
- `dim_feedforward=84`
- `val_all_windows=True`
- `lr=5e-4`
- `weight_decay=1e-4`

Current result on the fixed split:

- single seed `42`: test RMSE `14.7126`
- seeds `42,43,44`: mean test RMSE `15.0937`

This improves over the original paper-table defaults in the current codebase, but it is no
longer a strict Table 2 reproduction because the learning rate and unspecified regularization
were changed.

## Protocol Note

Most ablations so far reuse one fixed engine-level split for control. That is useful for
 debugging, but it is slightly stricter than the paper text, which only states that `80%` of
 engines are used for training and `20%` for validation. A closer-to-paper interpretation is
 to resample that split for each independent experiment.

The runner now supports both modes explicitly:

- fixed split control:
  `train_supervised.py --split-seed 42`
- paper-closer protocol:
  `train_supervised.py --resample-split-per-seed`
