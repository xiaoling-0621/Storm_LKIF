# Typhoon ERA5 Causal Baseline

This baseline supports configurable target definition, EOF fitting scope, and Liang causality scope.

## Core Options

- `target_variable`
  - `wind`: use `wind_like_feature` as the target
  - `pressure`: use `pressure_like_feature` as the target

- `target_mode`
  - `raw`: use the aligned target series directly
  - `delta`: use next-step change `target[t+1] - target[t]`
  - In `delta` mode, the setup is `ERA5(t) -> target(t+1) - target(t)`, so the target length becomes `T-1` and ERA5 features are truncated to the first `T-1` timestamps

- `eof_fit_scope`
  - `per_storm`: fit EOF/PCA separately for each storm
  - `global_selected`: fit EOF/PCA on all currently selected storms together, then transform each storm with the shared basis

- `causality_scope`
  - `per_storm`: run Liang causality separately for each storm
  - `global_segmented`: run a pooled Liang analysis where time differences are computed only inside each storm segment, never across storm boundaries

## Recommended Baseline

The most conservative baseline combination is:

- `target_variable = wind`
- `target_mode = raw`
- `eof_fit_scope = per_storm`
- `causality_scope = per_storm`

## Outputs

Output filenames include:

- `target_variable`
- `target_mode`
- `eof_fit_scope`
- `causality_scope`

This avoids overwriting results from different experiment settings.
