# PRD: NetSense-QoR — GNN-Based Logic Synthesis QoR Predictor

## 1. Overview

Build a deep learning model that predicts the quality of a digital circuit after logic synthesis (area and delay) from two inputs:
1. The circuit, represented as an And-Inverter Graph (AIG)
2. A synthesis recipe: a sequence of 20 optimization commands from the ABC tool (e.g. balance, rewrite, refactor, resub)

Running the real synthesis tool for every possible recipe is slow. The model predicts the result in milliseconds, so many recipes can be ranked quickly and only the best ones need to be run for real.

This is a solo university deep learning course project. It must also be strong enough to show on a resume and explain in a viva, so understanding matters as much as results.

## 2. Who I am and how to work with me (rules for Claude Code)

- I am an undergraduate (CSE, AI/ML). I know Python, basic ML (scikit-learn), and full-stack development. I am NEW to PyTorch, graph neural networks, and logic synthesis.
- My goal is to learn, not just finish. For every file or block of code you write:
  - Explain what it does and why, in plain language, before or after writing it.
  - Point out any concept I should understand for my viva (e.g. message passing, pooling, normalisation).
- Work in small steps. Finish and test one piece before starting the next. Ask me before moving to a new phase.
- Prefer simple, readable code over clever code. Add short comments explaining intent.
- Never delete, move, or modify anything inside `data/`. Treat it as read-only.
- Never commit `data/`, model checkpoints, `.env` files, or large files to git.
- Do not add "Co-Authored-By: Claude" or "Generated with Claude Code" lines to commits or PR descriptions.
- Ask before installing new libraries, and add every library you install to `requirements.txt`.
- If something about the dataset format is unclear, inspect the actual files and tell me what you found. Do not guess.
- If the original OpenABC repo code uses old library versions, port it to current versions rather than downgrading everything.
- When something fails, explain the cause before fixing it.

## 3. Goals

1. Build a working data pipeline for the OpenABC-D dataset in PyTorch Geometric.
2. Reproduce a baseline model (GCN graph encoder + CNN recipe encoder) with results in a similar range to the OpenABC-D paper.
3. Improve on the baseline with controlled experiments (one change at a time).
4. Evaluate honestly on designs the model has never seen, not just a random split.
5. Produce clear plots, a results table, and written findings suitable for a report and viva.

## 4. Non-goals (for this phase)

- Verifying the model's top-ranked recipes by running them through ABC. That is the next phase, after this PRD is complete.
- Building a web demo or UI.
- Downloading the full 1.4 TB raw dataset. Only the ML-ready version (~19 GB) or a subset of it is used.
- Reinforcement learning or generating new recipes.

## 5. Background

- **Logic synthesis** converts a hardware description into an optimised network of logic gates.
- **AIG (And-Inverter Graph):** every circuit expressed using only 2-input AND gates and inverters. Nodes are gates; edges are wires; some edges are inverted.
- **Synthesis recipe:** an ordered sequence of ABC commands. Different orders give different results on the same circuit.
- **Area proxy:** number of AND nodes in the final AIG. **Delay proxy:** number of logic levels (longest path), or the delay label provided in the dataset.
- **OpenABC-D dataset** (Chowdhury et al., NYU, arXiv:2110.11292): 29 open-source hardware designs, each synthesised with 1500 random recipes of length 20 using yosys-abc. Repo: https://github.com/NYU-MLDA/OpenABC

## 6. Data

### 6.1 Source
- Use the ML-ready dataset linked from the OpenABC README: PyTorch files for the initial AIGs, recipe encodings (`synthID2Vec.pickle`), target labels (pickled), and train/test split CSVs. Total ~19 GB.
- Original design AIGs are in the repo under `bench_openabcd`.

### 6.2 First step: inspect before building
Before writing the Dataset class, write a short exploration notebook that:
- Loads one graph file and prints its contents (node features `x`, `edge_index`, any edge attributes, number of nodes/edges).
- Identifies what each node-feature column means (cross-check with the README and paper).
- Loads `synthID2Vec.pickle`, prints one recipe, and maps its numbers back to ABC command names.
- Loads the labels and identifies the columns for final node count (area) and delay.
- Checks whether the same initial graph is stored once per design or repeated per recipe.
- Reports graph sizes per design and checks for NaNs or invalid values.

Summarise the findings in `docs/data_notes.md`. All later code must be based on these findings, not assumptions.

### 6.3 Subset strategy
- Start with 3 to 4 designs (choose small-to-medium graphs) for fast iteration.
- Scale to more designs only after the full pipeline works end to end.

### 6.4 Splits
Implement two split types and save each as a file in `splits/`:
1. **Random split:** recipes of the same designs split into train/val/test (e.g. 70/15/15). Tests generalisation to unseen recipes.
2. **By-design split:** whole designs held out for test. Tests generalisation to unseen circuits (the harder, more important case).
Also support **leave-one-design-out** as an option.

No sample may appear in more than one split. Add a test that checks this.

## 7. Model

### 7.1 Inputs and outputs
- Input: AIG graph (PyG `Data` object) + recipe (tensor of 20 integer command IDs).
- Output: 2 values: predicted area and predicted delay.

### 7.2 Targets
Support two target modes, selectable in config:
- `raw`: final node count and delay as-is (standardised).
- `ratio`: final value divided by the initial design's value (e.g. final nodes / initial nodes). This makes targets comparable across designs of very different sizes and is expected to generalise better.
Normalisation statistics must be computed on the training set only.

### 7.3 Baseline architecture
- **Graph encoder:** 2–3 GCN layers with ReLU, then global mean pooling → graph vector. (Mean pooling keeps the output independent of circuit size.)
- **Recipe encoder:** embedding layer (one learned vector per ABC command) → small 1D CNN over the 20 steps → pooling → recipe vector.
- **Head:** concatenate graph vector and recipe vector → 2-layer MLP → 2 outputs.
- Optimisation: Adam, lr 1e-3, MSE loss, early stopping on validation loss.

### 7.4 Planned variants (experiments)
Each experiment changes exactly one thing relative to the baseline:
1. Target: `raw` vs `ratio`
2. Graph encoder: GCN vs GIN vs GraphSAGE
3. Recipe encoder: 1D CNN vs small transformer encoder (2 layers, learned positional embeddings)
4. Loss: MSE vs MSE + pairwise ranking loss (for pairs of recipes on the same design, the recipe with the smaller true area should get the smaller prediction)
5. Ablation: graph-only vs recipe-only vs both
6. Generalisation: random split vs by-design split vs by-design + few-shot fine-tuning (fine-tune on 20 recipes of a held-out design, test on the rest)

All model choices must be selectable through a config file (YAML), not by editing code.

## 8. Evaluation

### 8.1 Metrics
- **MAPE** (mean absolute percentage error) for area and delay, computed on real-scale values after undoing normalisation.
- **Kendall's tau** per design (scipy.stats.kendalltau) between predicted and true area across that design's test recipes, then averaged across designs. This measures whether the model ranks recipes correctly, which is the real use case.
- Optionally: top-k accuracy (does the model's top 10 recipes overlap with the true top 10?).

### 8.2 Results table
Final table: rows = model variants, columns = MAPE (area, delay) and Kendall's tau, under random split / by-design split / by-design + few-shot.

### 8.3 Plots (saved to `results/`)
- Training and validation loss curves per run
- Predicted vs actual scatter plots (area and delay)
- Bar chart comparing experiments

## 9. Repository structure

```
netsense-qor/
├── CLAUDE.md              # project context and rules for Claude Code
├── PRD.md                 # this document
├── README.md              # project summary, how to run, results
├── requirements.txt
├── .gitignore             # must include data/, runs/, checkpoints/, .env
├── .claude/
│   └── settings.json      # Claude Code settings (attribution off, .env blocked)
├── configs/
│   ├── baseline.yaml
│   └── ...                # one config per experiment
├── data/                  # dataset (read-only, never committed)
├── docs/
│   ├── data_notes.md      # findings from data exploration
│   ├── data_sources.md    # where each data file came from + date
│   └── findings.md        # what each experiment showed and why
├── notebooks/
│   └── 01_explore.ipynb
├── splits/                # saved split files
├── src/
│   ├── data.py            # Dataset class, normalisation, splits
│   ├── models.py          # encoders and full model
│   ├── train.py           # training loop (CLI: --config path)
│   ├── evaluate.py        # metrics and evaluation
│   └── utils.py           # seeding, logging, plotting helpers
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── runs/                  # per-run logs and metrics CSVs (not committed)
└── results/               # final plots and tables (committed)
```

## 10. Environment and compute

- **Local machine:** development and debugging on CPU with a small data subset. Windows + WSL may be used; keep paths OS-independent (use pathlib).
- **Training:** free Kaggle GPU (~30 hrs/week). Code must run in a Kaggle notebook by: cloning the repo, `pip install -r requirements.txt`, and pointing a config value at the attached dataset path.
- Data paths must come from config or a command-line argument, never hardcoded.
- Libraries: Python 3.11, PyTorch, PyTorch Geometric, pandas, numpy, scipy, scikit-learn, matplotlib, tqdm, pyyaml, pytest.
- Set seeds (random, numpy, torch) for reproducibility.

## 11. Logging

- Each run writes to `runs/<experiment_name>_<timestamp>/`: the config used, per-epoch metrics CSV, and the best checkpoint.
- Print a short summary at the end of each run (best epoch, val/test metrics).

## 12. Milestones and acceptance criteria

**M1. Project setup**
- Repo structure created, `.gitignore` in place, `requirements.txt` installs cleanly.
- Done when: `pytest` runs (even with placeholder tests) and the repo is on GitHub without any data files.

**M2. Data exploration**
- Notebook and `docs/data_notes.md` completed as described in 6.2.
- Done when: I can explain the format of graphs, recipes, and labels from the notes.

**M3. Data pipeline**
- Dataset class returns (graph, recipe tensor of length 20, targets, design name).
- Both split types and both target modes work.
- Each design's initial graph is loaded once if the data allows it.
- Done when: tests pass for recipe length, command ID range, no NaNs, valid edge indices, and no overlap between splits; one DataLoader batch prints the expected shapes.

**M4. Baseline model**
- Model as in 7.3.
- Done when: a forward pass on one batch gives shape [batch, 2], and the model overfits 50 samples to near-zero training loss.

**M5. Training and evaluation**
- Training loop, early stopping, checkpointing, metrics from 8.1, logging from 11.
- Done when: the baseline trains on Kaggle on the subset, and MAPE, Kendall's tau, loss curves, and scatter plots are produced.

**M6. Experiments**
- All variants in 7.4 run via configs, one change at a time.
- Done when: every experiment has a row in the results table and a short entry in `docs/findings.md`.

**M7. Generalisation**
- By-design split, optional leave-one-design-out, and few-shot fine-tuning.
- Done when: the final results table in 8.2 is complete and saved in `results/`.

**M8. Wrap-up**
- README with project summary, how to run, final results table, and key plots.
- Done when: someone new could clone the repo, follow the README, and reproduce a baseline run.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset format differs from what the paper describes | Inspect files first (M2); base all code on `docs/data_notes.md` |
| Old repo code incompatible with current PyTorch Geometric | Port to current versions; don't downgrade |
| CUDA / PyG version mismatches | Train on Kaggle, where PyTorch is preinstalled; install PyG to match its CUDA version |
| Memory issues with large graphs | Start with small designs; load each design graph once; tune batch size |
| Model memorises designs, fails on unseen ones | Ratio targets, mean pooling, ranking loss, more designs, few-shot fine-tuning; report the gap honestly |
| Running out of time | Milestones M1–M6 alone make a complete class project; M7 is the resume upgrade |

## 14. Deliverables

- Clean GitHub repo following section 9
- Trained baseline and best model checkpoints (stored outside git, e.g. Kaggle output)
- `results/`: plots and final results table
- `docs/findings.md`: explanation of each experiment
- README suitable for recruiters and examiners

## 15. Credit

This project uses the OpenABC-D dataset and builds on its baseline. Cite: Chowdhury, Tan, Karri, Garg, "OpenABC-D: A Large-Scale Dataset For Machine Learning Guided Integrated Circuit Synthesis", arXiv:2110.11292. The README must clearly separate what comes from the original work and what is my own contribution (pipeline port, ratio targets, transformer encoder, ranking loss, generalisation study).

## 16. How to start

Begin with Milestone M1 only. Show me the planned folder structure first, explain each part, and wait for my go-ahead before creating files.
