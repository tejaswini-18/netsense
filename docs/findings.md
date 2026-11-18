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

## M6 experiments

**Setup:** every config in `configs/` (baseline + 8 variants), each trained with seeds 0, 1 and 2 = 27 runs,
on a Kaggle T4 GPU (about 55 minutes in total; notebook `notebooks/03_kaggle_experiments.ipynb`, code commit
`fec437c`). Every variant differs from the baseline in exactly **one** setting. The train/val/test split is
fixed by `split_seed: 42`, so all runs see the same samples; only the model's starting weights and the
shuffling order change with the seed. No run failed.

Files in `results/experiments/`: `results_table.md` / `.csv` (mean ± std over seeds), `all_runs.csv`
(one row per run), `experiments_bar.png` (tau with error bars), `training_log.txt` (every epoch of every
run) and `kaggle_notebook.ipynb` (the executed notebook).

| Experiment (one change from baseline) | tau AND | tau delay | MAPE AND % | MAPE delay % | top-10 AND | top-10 delay |
|---|---|---|---|---|---|---|
| baseline | 0.107 ± 0.114 | 0.087 ± 0.023 | 2.07 ± 0.17 | 8.48 ± 0.19 | 0.025 ± 0.020 | 0.067 ± 0.031 |
| exp1_ratio: ratio targets | **0.461 ± 0.018** | **0.137 ± 0.034** | **0.72 ± 0.04** | 7.94 ± 0.22 | **0.167 ± 0.031** | **0.167 ± 0.042** |
| exp2_gin: GIN graph encoder | 0.210 ± 0.018 | 0.093 ± 0.022 | 2.60 ± 0.79 | 8.19 ± 0.21 | 0.083 ± 0.012 | 0.117 ± 0.042 |
| exp2_sage: GraphSAGE encoder | 0.093 ± 0.074 | 0.056 ± 0.038 | 3.11 ± 0.64 | 7.99 ± 0.25 | 0.100 ± 0.000 | 0.083 ± 0.031 |
| exp3_transformer: transformer recipe encoder | 0.257 ± 0.052 | 0.133 ± 0.021 | 2.40 ± 0.32 | 8.04 ± 0.37 | 0.075 ± 0.041 | 0.092 ± 0.047 |
| exp4_rankloss: MSE + ranking loss | 0.338 ± 0.025 | 0.135 ± 0.016 | 5.47 ± 0.22 | 9.13 ± 0.42 | 0.117 ± 0.047 | 0.158 ± 0.012 |
| exp5_graph_only: no recipe input | 0.003 ± 0.042 | -0.001 ± 0.006 | 2.07 ± 0.26 | 7.59 ± 0.10 | 0.042 ± 0.024 | 0.025 ± 0.000 |
| exp5_recipe_only: no graph input | -0.006 ± 0.018 | 0.103 ± 0.004 | 66.91 ± 0.01 | 56.15 ± 0.09 | 0.017 ± 0.012 | 0.083 ± 0.012 |
| extra_select_tau: best epoch by val tau | 0.176 ± 0.083 | 0.094 ± 0.004 | 6.24 ± 2.94 | 15.09 ± 8.84 | 0.050 ± 0.020 | 0.067 ± 0.012 |
| design average (reference, ignores recipe) | — | — | 1.47 | 7.42 | — | — |

Tau is Kendall's tau averaged over the 4 designs (higher is better; 0 = no ranking ability). Top-10 overlap
by pure chance is about 0.044. "± std" is the spread over the 3 seeds (population standard deviation).
With only 3 seeds the std itself is rough, so small differences (a few hundredths of tau) are not reliable.

**What each experiment showed**

1. **Ratio targets are the clear winner (experiment 1).** AND tau rose from 0.11 to 0.46, with a much
   smaller spread (± 0.02 vs ± 0.11), and every design improved in every seed (e.g. fir 0.47–0.58,
   wb_dma 0.27–0.34). It is also the **first model to beat the design-average reference** on AND MAPE
   (0.72% vs 1.47%). Why: dividing by the design's starting size removes the between-design scale that
   held 99.7% of the AND variance, so the loss is now mostly about the recipe effect, which is exactly what
   we want to learn. Delay improved less (tau 0.09 → 0.14), and delay MAPE (7.94%) is still slightly worse
   than the reference (7.42%). Delay is the harder target: even the reference guess has 7.4% error, and only 11% of its variance comes from the recipe.
2. **GNN type (experiment 2): GIN helps a little, GraphSAGE does not.** GIN's AND tau is 0.21 ± 0.02,
   versus 0.11 for GCN. GIN uses a sum and an MLP per layer, so it can tell apart neighbourhoods that GCN's
   averaging treats as the same (it is as powerful as the Weisfeiler-Lehman test). GraphSAGE was no better
   than GCN. Given the baseline's large spread, the GIN gain is suggestive rather than proven.
3. **The transformer recipe encoder helps (experiment 3).** AND tau 0.26 ± 0.05 and delay tau 0.13 ± 0.02,
   both above the CNN baseline. Self-attention lets every step of the recipe look at every other step, and
   the order of synthesis passes matters. A CNN with kernel 3 only sees nearby steps in each layer.
4. **The ranking loss improves ranking but hurts accuracy (experiment 4).** AND tau 0.34 and the best
   delay top-10 among the raw-target runs, but AND MAPE rose to 5.5%. The ranking term only cares about the
   *order* within a design, not the actual values, so it pulls predictions away from the true scale. The
   runs also stopped early (best epoch 7–8 in all seeds), because validation MSE, which still picks the
   epoch, gets worse as the ranking term takes over. This is a trade-off you set with `rank_weight`.
5. **Ablations (experiment 5): both inputs are needed, and they do different jobs.**
   - *Graph only:* tau ≈ 0. Without the recipe, every recipe of a design gets the same input and therefore
     the same prediction, so the model cannot rank recipes at all. (The tiny non-zero taus come from GPU
     rounding noise.) Its MAPE (2.07% / 7.59%) is close to the design-average reference: **the graph branch
     learns "which design is this and how big is it"**.
   - *Recipe only:* MAPE 67% / 56%. Without the graph it cannot tell the designs apart and predicts one
     value for all of them. AND tau ≈ 0, but delay tau 0.10 is as good as the full baseline. So **some
     recipe patterns improve delay on every design** (likely `balance`, which ABC uses to reduce logic depth), while
     the AND effect of a recipe depends on the circuit.
6. **Choosing the epoch by validation tau backfired (extra experiment).** For seeds 0 and 1, validation tau
   happened to be highest at epoch 1 or 2, before the model had learned the target scale (validation delay
   MAPE 28% at epoch 1). Tau then did not beat that early peak for 10 epochs, so early stopping ended the
   run. Seed 2 trained to epoch 55 and reached AND tau 0.29. Result: high spread, poor MAPE. Validation tau
   on 225 recipes per design is too noisy to choose epochs on its own. A fix would be a minimum number of
   epochs, or choosing on a combination of loss and tau. This was not tested.

**What this means**

- The ranking ability of the baseline was limited mainly by the **target scaling**, not by the model's
  architecture. Fixing the target (ratio) did more than any change to the encoder, the loss or the epoch choice.
- Each improvement was measured on its own. Combining the ones that helped (ratio targets + GIN +
  transformer, perhaps with a small ranking-loss weight) is the obvious next configuration. It is not in
  the PRD's experiment list, so it is a choice for later.
- All of this uses a random split, so the test recipes belong to designs the model has trained on.
  Whether the model generalises to **unseen designs** is the question for M7 (by-design and
  leave-one-design-out splits).

## M7 generalisation to unseen designs

**Question:** can the model rank recipes for a circuit it has never seen? (PRD experiment 6, final table §8.2.)

**Setup** (`configs/m7_generalisation.yaml`, `src/run_generalisation.py`, notebook
`notebooks/04_kaggle_generalisation.ipynb`, code commit `75214df`, Kaggle T4 GPU):

- **12 designs:** the 4 used so far plus iir, tv80, Rocket_csr, Rocket_muldiv, or1200_multmac, s38417, s38584,
  sha256. All are under ~19k nodes and their labels really vary with the recipe. Designs whose labels barely
  change were left out (s35932 has only 5 distinct AND values; des3_area, ac97_ctrl, pci and dynamic_node vary
  by under 2%).
- **3 variants:** `m7_baseline` (raw targets, GCN, CNN), `m7_ratio` (the M6 winner) and `m7_combined`
  (ratio + GIN + transformer, the changes that helped in M6, together).
- **3 settings**, each with seeds 0, 1, 2:
  - *random:* recipes split at random, every design seen in training (as in M6). 225 test recipes per design.
  - *unseen:* by-design 4-fold split. Each fold's model never sees 3 designs; every design is held out once
    and the 4 folds' predictions are pooled. Scored on 1,480 recipes per design.
  - *few_shot:* each fold model, fine-tuned for 100 steps (lr 1e-3) on 20 fixed recipes of one held-out design,
    then scored on that design's other 1,480 recipes: exactly the same recipes as *unseen*.
- **Speed:** the dataset now encodes each design's graph once per batch, not once per sample, which gives
  identical predictions (tested) and is 13× faster on the laptop with 4 designs. Batch size 64 (not 32) for all
  M7 runs, because each batch encodes all its distinct design graphs, so bigger batches mean less graph work.
- **Reference guesses** (they ignore the recipe, so they cannot rank; MAPE only):
  - random: each design's training average
  - unseen: the training designs' average final/initial ratio × the new design's initial value
  - few-shot: the average of the design's 20 shot recipes

All 3 × 3 × (1 + 4) = 45 training runs and all fine-tunings finished. Files are in `results/generalisation/`:
`results_table.md`/`.csv`, `all_runs.csv`, `per_design.csv` (tau per design), `generalisation_bar.png`,
`training_log.txt` and `kaggle_notebook.ipynb`.

| Variant | Setting | tau AND | tau delay | MAPE AND % | MAPE delay % |
|---|---|---|---|---|---|
| m7_baseline | random | 0.093 ± 0.016 | 0.053 ± 0.014 | 3.00 ± 1.37 | 32.4 ± 4.3 |
| m7_ratio | random | **0.271 ± 0.032** | **0.083 ± 0.011** | 0.98 ± 0.06 | 22.5 ± 2.0 |
| m7_combined | random | 0.188 ± 0.024 | 0.072 ± 0.007 | 0.99 ± 0.08 | 20.4 ± 5.7 |
| reference: design's training average | random | — | — | **0.92** | **11.2** |
| m7_baseline | unseen | 0.057 ± 0.003 | -0.018 ± 0.008 | 174.6 ± 3.9 | 1056 ± 66 |
| m7_ratio | unseen | **0.184 ± 0.010** | -0.014 ± 0.007 | 16.4 ± 0.2 | 304 ± 25 |
| m7_combined | unseen | 0.135 ± 0.006 | -0.004 ± 0.016 | 13.9 ± 0.3 | 198 ± 10 |
| reference: average ratio of training designs | unseen | — | — | **10.8** | **128** |
| m7_baseline | few_shot | 0.085 ± 0.007 | 0.004 ± 0.011 | 3.35 ± 0.33 | 33.7 ± 16.5 |
| m7_ratio | few_shot | 0.202 ± 0.028 | 0.023 ± 0.010 | 1.07 ± 0.17 | 24.2 ± 1.9 |
| m7_combined | few_shot | **0.224 ± 0.015** | **0.056 ± 0.004** | **0.85 ± 0.03** | 17.5 ± 1.7 |
| reference: average of the 20 shots | few_shot | — | — | 0.96 | **10.9** |

Top-10 overlap is in the results files. It is near chance everywhere outside the random split (chance is
10/1480 ≈ 0.007 for unseen/few-shot and 10/225 ≈ 0.044 for random), so it is left out here.

**What this shows**

1. **AND ranking partly transfers to unseen circuits, but only with ratio targets.** With ratio targets, tau
   drops from 0.27 on seen designs to 0.18 on unseen ones: it is lower, but clearly above zero and stable across
   seeds (± 0.01). It works for most held-out designs (Rocket_csr 0.30, s38417 0.33, fir 0.29, tv80 0.28) but
   not for i2c, or1200_multmac and sha256 (≈ 0). With raw targets (m7_baseline) tau on unseen designs is only
   0.06, and the AND MAPE is 175%: a raw-target model has only learned the sizes of its training designs and
   has no idea of a new design's scale. Ratio targets express the result relative to the design's own starting
   size, which is known for any new circuit. This is the main reason the ratio idea matters.
2. **Predicting the *level* of an unseen design is still poor.** Even with ratio targets, AND MAPE on unseen
   designs (14–16%) is worse than the simple reference (10.8%). The model ranks recipes better than chance but
   cannot say where a new design's results will lie.
3. **Delay does not transfer at all.** Delay tau on unseen designs is about 0 for every variant, and delay MAPE
   is in the hundreds of %. The cause is in the data: final delay divided by the initial logic depth varies
   more than 20-fold between designs (about 24 for fir and iir, 150 for Rocket_csr, 549 for sha256, whose delay
   ranges from 10k to 155k), while final/initial AND only ranges from 0.62 to 0.94. So for delay, "ratio"
   does not put the designs on a common scale, and the training designs say little about a new one. Even
   the reference guess is off by 128%, and it is not one outlier: removing sha256 does not lower it (132%).
   Even on seen designs, every model's delay MAPE (20–32%) is worse than the design-average guess (11%).
   Delay ranking on seen designs is weak too (tau 0.05–0.08), except for sha256 (≈ 0.49 in all variants),
   whose delay varies hugely with the recipe.
4. **Few-shot fine-tuning on 20 recipes fixes most of the level error, and helps ranking for the combined
   model.**
   - *Level:* AND MAPE falls from 16.4% to 1.1% (ratio) and from 13.9% to 0.85% (combined). The 20 shots
     tell the model where the new design's results lie.
   - *Ranking:* tau improves a little for ratio (0.18 → 0.20) and clearly for combined (0.135 → 0.224).
   - *Reference:* combined + few-shot is the only model that beats the "average of the 20 shots" reference
     on AND MAPE (0.85% vs 0.96%), as well as ranking (which the reference cannot do at all). No model beats
     the references on delay MAPE.
5. **The M6 gains do not simply add up.** On seen designs, combined (ratio + GIN + transformer) ranks worse
   than ratio alone (0.19 vs 0.27). In M6, GIN and the transformer were each tested with raw targets on
   4 designs; with ratio targets on 12 designs the extra capacity did not help. After fine-tuning, combined
   is the best model (0.224), so its larger encoders may adapt better from a few examples. With 3 seeds this
   is a hint, not a proof.

**Caveats**
- The random-split tau is measured on 225 recipes per design, the others on 1,480 (the same recipes for unseen
  and few-shot), so random vs unseen is a slightly different test.
- In the by-design split, early stopping uses validation recipes of the *training* designs, so it does not
  know how well the model does on unseen designs.
- The fine-tuning settings (100 steps, lr 1e-3) were fixed in advance, not tuned. A held-out design has no
  validation data to tune them on.
- 12 training designs is still few. The OpenABC-D paper trains on many more, and generalisation would likely
  improve with more designs.
- The M7 random-split numbers differ from M6 (e.g. ratio tau 0.27 vs 0.46) because the design set
  (12 vs 4) and batch size (64 vs 32) changed. M7 rows are comparable with each other, not with M6.

**Bug found on Kaggle (fixed after the run):** the quick test cell reported 1 failed test
(`test_runner_builds_table_and_keeps_the_split_fixed_across_seeds`). After the batching speed-up, a 1-epoch
test run takes under a second on the GPU. Two runs with the same name could start within the same second, and
run folders were named by the time to the second, so the second folder already existed. The real experiment
runs take minutes and were not affected (all 27 settings finished). Fix: `make_run_dir` adds `_2`, `_3`, …
when the folder exists, with a new test.
