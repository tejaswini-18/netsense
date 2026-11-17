# Findings

What each experiment showed and why. Numbers are on the test set unless stated otherwise.

## Baseline (M5)

**Setup:** `configs/baseline.yaml`. 4 designs (i2c, spi, fir, wb_dma), random split by recipe
(4200 / 900 / 900 samples), raw targets (final AND count, delay) standardised with training statistics,
3-layer GCN + mean pooling, embedding + 1D CNN recipe encoder, 2-layer MLP head, Adam lr 1e-3, MSE loss.
Trained twice on a Kaggle GPU (torch 2.11, about 5 s per epoch vs about 41 s on the laptop CPU) with
identical code and seed. Run 2 (`baseline_20261117_123717`, Kaggle notebook version 1) is kept in
`results/baseline/`: test metrics, both plots, the full training log and the executed notebook.

**Training (run 2):** best epoch 18 (validation loss 0.055); early stopping ended training at epoch 28.
Validation loss levelled off from about epoch 5 while training loss kept falling: the model started to
overfit, and early stopping kept the epoch-18 weights.

| Metric | Run 1 | Run 2 | Guess each design's training average (ignores the recipe) |
|---|---|---|---|
| MAPE, AND | 1.69% | 2.30% | **1.47%** |
| MAPE, delay | 8.07% | 8.87% | **7.42%** |
| Kendall's tau, AND (mean over designs) | 0.137 | 0.148 | — (no ranking) |
| Kendall's tau, delay | 0.080 | 0.063 | — |
| Top-10 overlap, AND (pure chance: 10/225 = 0.044) | 0.075 | 0.150 | — |
| Top-10 overlap, delay | 0.050 | 0.025 | — |

Kendall's tau by design (run 2): AND: fir 0.200, i2c -0.158, spi 0.408, wb_dma 0.143;
delay: fir -0.007, i2c -0.005, spi 0.098, wb_dma 0.166.

**Run-to-run variation.** Same code, same seed, different results: on a GPU, operations such as summing
neighbour messages run in parallel and add numbers in a varying order, so floating-point rounding differs
and the trained model drifts. (Two CPU runs on the laptop matched exactly.) The variation is large:
0.6 points of AND MAPE and a factor of 2 in top-10 overlap. **M6 must compare experiments over several
seeds and report mean ± spread, not single runs.**

**What this shows**

1. **The low MAPE is misleading.** A predictor that ignores the recipe and outputs each design's average
   has a lower MAPE than the model. Most of the error budget is about *which design*, not *which recipe*.
2. **Why: almost all target variance is between designs.** In the training data, 99.7% of the variance of
   the AND count (89% for delay) comes from differences between designs; only 0.3% (11%) comes from
   different recipes on the same design. With raw targets standardised over all designs, the MSE loss is
   dominated by getting each design's level right, and the recipe effect, which is what we want to rank,
   is a tiny part of the training signal.
3. **Ranking is weak but not zero.** Mean tau about 0.14 (AND) and 0.06–0.08 (delay), negative for i2c in
   both runs. Top-10 overlap is close to chance. The scatter plots agree: each design forms a tight cluster in
   the right place, but within a cluster the predictions barely follow the true values.

4. **Early stopping on MSE stops before ranking peaks.** In run 2, validation tau (AND) rose from 0.185 at
   the chosen epoch 18 to 0.268 at epoch 28, while validation loss drifted up. The MSE loss mostly measures
   design-level accuracy, so it is a poor guide to ranking quality.

**Implications for M6** (each is a planned experiment from PRD §7.4):
- `ratio` targets (experiment 1) remove most of the between-design scale, so more of the loss is about recipes.
- A pairwise ranking loss (experiment 4) trains directly on "which recipe is better for this design".
- The recipe-only ablation (experiment 5) will show how much the graph encoder adds within a design.
- The design-average guess should be reported as a reference row in the results table.
- Consider choosing the best epoch by validation tau instead of validation loss (a config option), since
  ranking is the real goal. This is a change to model selection, so it gets its own controlled comparison.
