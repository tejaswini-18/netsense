# Findings

What each experiment showed and why. Numbers are on the test set unless stated otherwise.

## Baseline (M5)

**Setup:** `configs/baseline.yaml`. 4 designs (i2c, spi, fir, wb_dma), random split by recipe
(4200 / 900 / 900 samples), raw targets (final AND count, delay) standardised with training statistics,
3-layer GCN + mean pooling, embedding + 1D CNN recipe encoder, 2-layer MLP head, Adam lr 1e-3, MSE loss.
Trained on a Kaggle GPU (run `baseline_20261117_123036`).

**Training:** best epoch 19 (validation loss 0.054); early stopping ended training at epoch 29.
Validation loss levelled off from about epoch 5 while training loss kept falling: the model started to
overfit, and early stopping kept the epoch-19 weights.

| Metric | AND count | Delay |
|---|---|---|
| MAPE (model) | 1.69% | 8.07% |
| MAPE (guess each design's training average, ignore the recipe) | **1.47%** | **7.42%** |
| Kendall's tau, mean over designs | 0.137 | 0.080 |
| Top-10 overlap (pure chance: 10/225 = 0.044) | 0.075 | 0.050 |

Kendall's tau by design: AND: fir 0.169, i2c -0.174, spi 0.407, wb_dma 0.144;
delay: fir -0.033, i2c -0.005, spi 0.157, wb_dma 0.199.

**What this shows**

1. **The low MAPE is misleading.** A predictor that ignores the recipe and outputs each design's average
   has a lower MAPE than the model. Most of the error budget is about *which design*, not *which recipe*.
2. **Why: almost all target variance is between designs.** In the training data, 99.7% of the variance of
   the AND count (89% for delay) comes from differences between designs; only 0.3% (11%) comes from
   different recipes on the same design. With raw targets standardised over all designs, the MSE loss is
   dominated by getting each design's level right, and the recipe effect, which is what we want to rank,
   is a tiny part of the training signal.
3. **Ranking is weak but not zero.** Mean tau 0.14 (AND) and 0.08 (delay), negative for i2c. Top-10
   overlap is only slightly above chance. The scatter plots agree: each design forms a tight cluster in
   the right place, but within a cluster the predictions barely follow the true values.

**Implications for M6** (each is a planned experiment from PRD §7.4):
- `ratio` targets (experiment 1) remove most of the between-design scale, so more of the loss is about recipes.
- A pairwise ranking loss (experiment 4) trains directly on "which recipe is better for this design".
- The recipe-only ablation (experiment 5) will show how much the graph encoder adds within a design.
- The design-average guess should be reported as a reference row in the results table.
