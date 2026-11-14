# Data notes

Findings from inspecting the OpenABC-D files. All later code must be based on these notes, not assumptions.
Status: **complete for M2**. Every fact below was checked on the real files (see `notebooks/01_explore.ipynb`)
unless it says it comes from the OpenABC source code.

Dataset location: `OPENABC2_DATASET/` in the project root (read-only, gitignored). 18 GB, 63,030 files.

## Folder layout

| Path | What it is |
|---|---|
| `processed/<design>_syn<N>_step0.pt.zip` | 63,000 graph files = 42 designs × 1500 recipes (N = 0–1499) |
| `processed/pre_filter.pt`, `pre_transform.pt` | 431-byte PyG housekeeping files; not needed |
| `synthesisStatistics.pickle` | Labels (see below) |
| `origAIGNodesDict.pickle` | Dict of 29 designs → initial AND count (see Labels) |
| `train_/test_data_*.csv` | Split files: one column `fileName`, listing graph file names. No labels inside |

Every graph file is `step0`: the circuit **before** synthesis. The model gets this plus the recipe and
predicts the result after all 20 steps.

## Graph files

- Each `.pt.zip` is a zip containing one `torch.save()` file of a PyTorch Geometric `Data` object.
- Two save formats:
  - **41 designs: PyTorch Geometric 1.7** (fields stored directly, node count in `__num_nodes__`).
    The OpenABC README warns PyG 2.x cannot read these directly.
  - **`xbar` only: PyG 2.x** (fields inside a `GlobalStorage` object, node count in `num_nodes`).
- **Port:** `src/data.py::load_aig_file` unzips in memory, unpickles with an allowlist of classes (anything
  else is refused, so a file cannot run code), and rebuilds a current PyG 2.8 `Data`. Handles both formats.
- Fields (after loading):
  - Per node: `node_type`, `num_inverted_predecessors`, `node_id` (gate names such as `new_n211_`)
  - Per edge: `edge_index`, `edge_type`
  - Per graph: `longest_path`, `and_nodes`, `pi`, `po`, `not_edges`, `desName`, `synVec`, `synID`, `stepID`
  - `x`, `edge_attr`, `y`, `pos`, `normal`, `face` exist but are `None`. **There is no ready-made feature
    matrix `x`;** we build node features from `node_type` and `num_inverted_predecessors` in M3.
  - All tensors are int64. Every design: `num_nodes == and_nodes + pi + po`; all edge indices in range.
- Meanings (from OpenABC `PyGDataAIG.py`, then checked on the data):
  - `node_type`: 0 = PI (primary input), 1 = PO (primary output), 2 = AND gate
  - `edge_type`: 1 = inverted wire (NOT), 0 = normal wire
  - `num_inverted_predecessors`: number of inverted incoming wires (0–2). Verified: equals the count of
    `edge_type == 1` edges arriving at each node.
- **Edge direction: `edge_index[0]` = receiving gate, `edge_index[1]` = source.** Verified: counting incoming
  wires via row 0 gives PI = 0, PO = 1, AND = 2 on every design checked (ss_pcm, aes, xbar).
  PyG's default message flow is row 0 → row 1, so as stored, messages would travel from outputs back to inputs.
  **M3 must flip the edges (or add reverse edges).**
- **The graph is identical in all 1500 files of a design**; only `synVec`/`synID` differ (checked syn0 vs
  syn1, 777, 1499 on ss_pcm, i2c, aes). So each design's graph is loaded once.

## Recipes

`synVec` holds 20 integers, one per ABC command, in order. Mapping (OpenABC `synthID2SeqMapping.py`):

| ID | ABC command |
|---|---|
| 0 | refactor |
| 1 | refactor -z |
| 2 | rewrite |
| 3 | rewrite -z |
| 4 | resub |
| 5 | resub -z |
| 6 | balance |

- All 42 designs: `synVec` has length 20 and values 0–6.
- **Recipe N is the same for every design** (checked syn0 and syn42 across 4 designs).
  Example, recipe 0: resub -z; rewrite -z; refactor; rewrite; rewrite; balance; refactor; resub; ...
- `synthID2Vec.pickle` (mentioned in the PRD) is **not** in this download, but it is not needed: each graph
  file already contains its `synVec`.

## Labels

`synthesisStatistics.pickle`: dict, 42 designs → list of 5 lists, each of length 1500 (one value per recipe,
indexed by synID). Contains only plain Python data, so it is safe to load. Order (OpenABC `pickleStatsForML.py`):

| Index | Meaning | Type | Example (ss_pcm, syn0) |
|---|---|---|---|
| 0 | AND-gate count of final AIG ("nodes") | int | 395 |
| 1 | NOT count of final AIG | int | 473 |
| 2 | Longest path (logic levels) of final AIG | int | 8 |
| 3 | Area after technology mapping | float | 293.93 |
| 4 | Delay after technology mapping | float | 520.76 |

- The original training code uses index 0 ("nodes"), 3 (area) or 4 (delay) as targets.
- **Our choice (decide in M3):** area target = index 0 (AND count, as in PRD §5) or index 3 (mapped area);
  delay target = index 4 (or index 2 as a proxy).
- Original code normalises per design with mean/std over **all 1500 recipes, including test ones**. That leaks
  test information. We will compute statistics on the training set only (PRD §7.2).

- **Small designs barely change in AND count:** ss_pcm's final AND count takes only 4 distinct values (395–398)
  across 1500 recipes, while its mapped `area` takes 120. Ranking recipes by AND count is nearly impossible for
  tiny designs. This is relevant for Kendall's tau and for choosing the area target.
- **`xbar` labels are degenerate:** final AND is 10,512 for all 1500 recipes, and the other 4 columns take only
  2 distinct values each. **Exclude `xbar` from training and evaluation.**

`origAIGNodesDict.pickle`: 29 designs → initial AND count. It matches the graph's `and_nodes` for 26 of the 27
designs present. Mismatch: `mem_ctrl` (graph 18,092 vs dict 16,307). `dft` and `idft` are in the dict but not
in the dataset. **For `ratio` targets, use the graph's own `and_nodes`, not this dict.**

## Splits

| Files | Train designs | Test designs | Overlap | Type |
|---|---|---|---|---|
| `set1` | 27 | 27 | all 27 | Random/by recipe (README: first 1000 recipes train, next 500 test) |
| `set2` | 18 | 9 | none | **By design** (unseen circuits) |
| `mixmatchv1` (= `set3` in code) | 27 | 27 | all 27 | Mixed, by recipe |
| `ncsu_set1` | 27 | 15 | none | By design |

- `*_area_*` CSVs point to `step20` files that are **not** in this download. Ignore them.
- `set1`/`set2` use the same 27 designs; the other 15 (e.g. picorv32, or1200, Rocket) appear only in the `ncsu` splits.

## Graph sizes

Sizes range from 597 nodes (`ss_pcm`) to 139,719 (`vga_lcd`). Full table: notebook section 6.

| Design | Nodes | Edges | AND | PI | PO | Depth | Final AND range |
|---|---|---|---|---|---|---|---|
| ss_pcm | 597 | 896 | 403 | 104 | 90 | 8 | 395–398 |
| usb_phy | 709 | 1,064 | 487 | 132 | 90 | 10 | 418–440 |
| sasc | 873 | 1,351 | 613 | 135 | 125 | 9 | 583–597 |
| simple_spi | 1,226 | 1,992 | 930 | 164 | 132 | 12 | 785–824 |
| i2c | 1,474 | 2,466 | 1,169 | 177 | 128 | 15 | 883–974 |
| spi | 4,711 | 8,676 | 4,219 | 254 | 238 | 35 | 2,952–3,280 |
| des3_area | 5,338 | 10,006 | 4,971 | 303 | 64 | 30 | 4,576–4,656 |
| fir | 5,319 | 9,467 | 4,558 | 410 | 351 | 47 | 3,903–4,083 |
| ... | | | | | | | |
| jpeg | 124,522 | 234,331 | 114,771 | 4,962 | 4,789 | 40 | 102,296–106,491 |
| vga_lcd | 139,719 | 227,731 | 105,334 | 17,322 | 17,063 | 23 | 103,534–104,232 |

Outlier: `fpu` has depth 819 (others ≤ 86).

**Subset suggestion for M3 (PRD §6.3):** small-to-medium designs with real spread in their labels, e.g.
`i2c`, `spi`, `fir`, `wb_dma` (all in the `set1`/`set2` designs). Avoid `ss_pcm`, `usb_phy` and `sasc` as the
main subset: their final AND counts barely vary.

## Data quality

- No NaN or negative values in any label (all 42 designs × 5 columns × 1500).
- Every label list has length 1500.
- All edge indices are within `[0, num_nodes)`.
- Only known problem: `xbar` (degenerate labels, different file format).
