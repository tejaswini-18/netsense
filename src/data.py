"""OpenABC-D data pipeline: graph loading, recipes, labels, splits, targets and the Dataset class.

Every format fact used here is documented (and was checked on the real files) in docs/data_notes.md.
"""

import io
import json
import pickle
import random
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Constants (see docs/data_notes.md)
# ---------------------------------------------------------------------------

NUM_RECIPES = 1500        # recipes per design, synID 0..1499
RECIPE_LENGTH = 20        # ABC commands per recipe
NUM_COMMANDS = 7          # command IDs 0..6
ABC_COMMANDS = {0: "refactor", 1: "refactor -z", 2: "rewrite", 3: "rewrite -z",
                4: "resub", 5: "resub -z", 6: "balance"}

# Column of each label inside synthesisStatistics.pickle
LABEL_COLUMNS = {"and": 0, "not": 1, "levels": 2, "area": 3, "delay": 4}

# For "ratio" targets: the property of the *initial* graph that each label is divided by.
# Only properties of the input graph are used, so ratios also work for designs never seen in training.
RATIO_BASE = {"and": "and_nodes", "not": "not_edges", "levels": "longest_path",
              "area": "and_nodes", "delay": "longest_path"}

# xbar's labels barely change across recipes (data_notes.md), so it must not be used
EXCLUDED_DESIGNS = {"xbar"}

NUM_NODE_FEATURES = 6     # one-hot node type (3) + one-hot inverted-input count (3)


class _OldStorage:
    """Stand-in for PyG 2.x GlobalStorage (used by xbar files): fields live in '_mapping'."""

    def __setstate__(self, state):
        self.fields = dict(state["_mapping"])


class _OldData:
    """Stand-in for the saved PyG Data class: it just collects the saved fields into a dict.

    PyG 1.7 files (41 designs) store fields directly; PyG 2.x files (xbar) wrap them in '_store'.
    """

    def __setstate__(self, state):
        if "_store" in state:
            self.fields = state["_store"].fields
        else:
            self.fields = dict(state)


class _SafeUnpickler(pickle.Unpickler):
    """Only allow the few classes these files need; anything else is refused.

    A normal pickle can import and run any code. Restricting it keeps a bad or corrupted file harmless.
    """

    ALLOWED = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch_geometric.data.data", "Data"): _OldData,
        ("torch_geometric.data.storage", "GlobalStorage"): _OldStorage,
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return self.ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"Blocked unexpected class in data file: {module}.{name}")


class _SafePickleModule:
    """torch.load() takes a 'pickle module'; this one supplies our restricted Unpickler."""

    Unpickler = _SafeUnpickler
    load = staticmethod(pickle.load)


def load_aig_file(path):
    """Read one <design>_syn<N>_step<M>.pt.zip file and return a current-format PyG Data object.

    The file is unzipped in memory; nothing is written to the dataset folder.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as outer:
        raw = outer.read(outer.namelist()[0])  # the .pt file inside the zip

    old = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False,
                     pickle_module=_SafePickleModule)

    # Keep only real values; drop empty PyG defaults (x, y, pos, ...).
    # The node count is called "__num_nodes__" in PyG 1.7 files and "num_nodes" in PyG 2.x files.
    num_nodes = old.fields.get("__num_nodes__", old.fields.get("num_nodes"))
    fields = {k: v for k, v in old.fields.items()
              if v is not None and k not in ("__num_nodes__", "num_nodes")}
    data = Data(**fields)
    if num_nodes is not None:
        data.num_nodes = num_nodes
    return data


# ---------------------------------------------------------------------------
# Design graphs (loaded once per design)
# ---------------------------------------------------------------------------

def graph_file(root, design, syn_id):
    return Path(root) / "processed" / f"{design}_syn{syn_id}_step0.pt.zip"


def load_design_graph(root, design):
    """Load a design's initial AIG and turn it into model input.

    The graph is identical in all 1500 files of a design, so we read recipe 0's file only.
    Returns (graph, info): graph has x, edge_index, edge_attr; info holds initial-graph counts.
    """
    raw = load_aig_file(graph_file(root, design, 0))

    # Node features: which kind of node (PI / PO / AND) and how many inverted inputs it has
    node_type = F.one_hot(raw.node_type, num_classes=3)
    inverted = F.one_hot(raw.num_inverted_predecessors, num_classes=3)
    x = torch.cat([node_type, inverted], dim=1).float()

    # Stored edges are [receiving gate, source]; flip to [source, receiving gate] so that
    # PyG's default message flow (row 0 -> row 1) follows the signal from inputs to outputs.
    edge_index = raw.edge_index.flip(0)
    edge_attr = raw.edge_type.float().unsqueeze(1)  # 1 = inverted wire (unused by GCN, kept for later)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=raw.num_nodes)
    info = {"and_nodes": int(raw.and_nodes), "not_edges": int(raw.not_edges),
            "longest_path": int(raw.longest_path), "num_nodes": raw.num_nodes}
    return graph, info


# ---------------------------------------------------------------------------
# Recipes and labels
# ---------------------------------------------------------------------------

def load_recipes(root, source_design="ss_pcm"):
    """Return a LongTensor [1500, 20]: row N is recipe N as ABC command IDs.

    Recipe N is the same for every design (data_notes.md), so we read it from the smallest design's files.
    """
    recipes = torch.empty(NUM_RECIPES, RECIPE_LENGTH, dtype=torch.long)
    for syn_id in range(NUM_RECIPES):
        d = load_aig_file(graph_file(root, source_design, syn_id))
        assert int(d.synID[0]) == syn_id, f"file for recipe {syn_id} says synID={d.synID}"
        recipes[syn_id] = d.synVec
    return recipes


def load_labels(root):
    """Return {design: FloatTensor [1500, 5]} with label columns in LABEL_COLUMNS order.

    The pickle holds only plain Python lists (no classes), so loading it cannot run code.
    """
    with open(Path(root) / "synthesisStatistics.pickle", "rb") as f:
        stats = pickle.load(f)
    return {des: torch.tensor(cols, dtype=torch.float).T for des, cols in stats.items()}


# ---------------------------------------------------------------------------
# Splits
# A split is {"train": [(design, syn_id), ...], "val": [...], "test": [...]}.
# ---------------------------------------------------------------------------

def _split_recipe_ids(seed, fractions):
    """Shuffle recipe IDs 0..1499 with a fixed seed and cut them into consecutive parts."""
    ids = list(range(NUM_RECIPES))
    random.Random(seed).shuffle(ids)
    parts, start = [], 0
    for frac in fractions[:-1]:
        end = start + round(frac * NUM_RECIPES)
        parts.append(sorted(ids[start:end]))
        start = end
    parts.append(sorted(ids[start:]))
    return parts


def make_random_split(designs, seed=42, val_fraction=0.15, test_fraction=0.15):
    """Same designs everywhere; recipes are divided into train/val/test.

    We split recipe IDs (not individual samples), so a test recipe is unseen for every design.
    Tests generalisation to new recipes.
    """
    train_ids, val_ids, test_ids = _split_recipe_ids(
        seed, [1 - val_fraction - test_fraction, val_fraction, test_fraction])
    return {"train": [(d, i) for d in designs for i in train_ids],
            "val": [(d, i) for d in designs for i in val_ids],
            "test": [(d, i) for d in designs for i in test_ids]}


def make_by_design_split(designs, test_designs, seed=42, val_fraction=0.15):
    """Whole designs are held out for test (unseen circuits).

    Validation (used for early stopping) takes a share of recipes from the training designs.
    """
    missing = set(test_designs) - set(designs)
    if missing:
        raise ValueError(f"test designs {sorted(missing)} are not in the design list")
    train_designs = [d for d in designs if d not in test_designs]
    if not train_designs:
        raise ValueError("by-design split needs at least one training design")
    train_ids, val_ids = _split_recipe_ids(seed, [1 - val_fraction, val_fraction])
    return {"train": [(d, i) for d in train_designs for i in train_ids],
            "val": [(d, i) for d in train_designs for i in val_ids],
            "test": [(d, i) for d in test_designs for i in range(NUM_RECIPES)]}


def make_leave_one_design_out_splits(designs, seed=42, val_fraction=0.15):
    """One by-design split per design, each holding out a single design for test."""
    return {d: make_by_design_split(designs, [d], seed, val_fraction) for d in designs}


def save_split(split, path):
    """Save as {part: {design: [syn_ids]}}, which is compact and easy to read."""
    out = {}
    for part, samples in split.items():
        grouped = {}
        for design, syn_id in samples:
            grouped.setdefault(design, []).append(syn_id)
        out[part] = grouped
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


def load_split(path):
    data = json.loads(Path(path).read_text())
    return {part: [(d, i) for d, ids in grouped.items() for i in ids] for part, grouped in data.items()}


# ---------------------------------------------------------------------------
# Targets and normalisation
# ---------------------------------------------------------------------------

def target_base(info, target_names):
    """Per-design divisors for ratio targets, e.g. [initial AND count, initial logic depth]."""
    return torch.tensor([float(info[RATIO_BASE[t]]) for t in target_names])


def compute_targets(labels, info, syn_ids, target_names, mode):
    """Targets for one design's recipes: FloatTensor [len(syn_ids), len(target_names)].

    mode "raw":   the label values themselves (e.g. final AND count, delay)
    mode "ratio": label / property of the initial graph (e.g. final AND / initial AND)
    """
    cols = [LABEL_COLUMNS[t] for t in target_names]
    values = labels[syn_ids][:, cols]
    if mode == "raw":
        return values
    if mode == "ratio":
        return values / target_base(info, target_names)
    raise ValueError(f"unknown target mode {mode!r} (use 'raw' or 'ratio')")


def to_real_scale(targets, base, mode):
    """Undo compute_targets: turn (un-standardised) raw or ratio targets back into label units."""
    return targets * base if mode == "ratio" else targets


class TargetScaler:
    """Standardise each target column: (value - mean) / std.

    fit() must only ever see training targets, so nothing about val/test leaks into training.
    """

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, train_targets):
        self.mean = train_targets.mean(dim=0)
        self.std = train_targets.std(dim=0).clamp_min(1e-8)  # guard against a constant column
        return self

    def transform(self, targets):
        return (targets - self.mean) / self.std

    def inverse_transform(self, scaled):
        return scaled * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QoRDataset(torch.utils.data.Dataset):
    """One sample = (design graph, recipe, targets, design name).

    Graphs are shared per design (no copies). With torch_geometric.loader.DataLoader a batch has:
      x, edge_index, batch  - all graphs of the batch merged into one big disconnected graph
      recipe [B, 20]        - command IDs
      y      [B, T]         - standardised targets
      base   [B, T]         - ratio divisors (to convert predictions back to real units)
      design (list of B names), syn_id [B]
    """

    def __init__(self, samples, graphs, recipes, targets, bases):
        self.samples, self.graphs, self.recipes = samples, graphs, recipes
        self.targets, self.bases = targets, bases

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        design, syn_id = self.samples[i]
        g = self.graphs[design]
        return Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, num_nodes=g.num_nodes,
                    recipe=self.recipes[syn_id].unsqueeze(0), y=self.targets[i].unsqueeze(0),
                    base=self.bases[i].unsqueeze(0), design=design, syn_id=syn_id)


def make_split(cfg, seed):
    kind = cfg["split"]
    if kind == "random":
        return make_random_split(cfg["designs"], seed, cfg["val_fraction"], cfg["test_fraction"])
    if kind in ("by_design", "leave_one_design_out"):
        test_designs = cfg["test_designs"]
        if kind == "leave_one_design_out" and len(test_designs) != 1:
            raise ValueError("leave_one_design_out needs exactly one design in test_designs")
        return make_by_design_split(cfg["designs"], test_designs, seed, cfg["val_fraction"])
    raise ValueError(f"unknown split {kind!r}")


def build_datasets(cfg, seed=42):
    """Build train/val/test datasets from the 'data' section of a config.

    Returns (datasets, scaler, meta). The scaler is fitted on training targets only.
    """
    root = Path(cfg["root"])
    designs = cfg["designs"]
    bad = EXCLUDED_DESIGNS & set(designs)
    if bad:
        raise ValueError(f"{sorted(bad)} must not be used (degenerate labels, see docs/data_notes.md)")

    # The split is regenerated from the seed; the saved file is a record and a safety check
    split = make_split(cfg, seed)
    split_file = Path(cfg["split_file"])
    if split_file.exists():
        if load_split(split_file) != split:
            raise ValueError(f"{split_file} does not match the current config; delete it or rename split_file")
    else:
        save_split(split, split_file)

    graphs, infos = {}, {}
    for d in designs:
        graphs[d], infos[d] = load_design_graph(root, d)
    recipes = load_recipes(root, cfg.get("recipe_source", "ss_pcm"))
    labels = load_labels(root)

    target_names, mode = cfg["targets"], cfg["target_mode"]

    def targets_for(samples):
        t = torch.stack([compute_targets(labels[d], infos[d], [i], target_names, mode)[0] for d, i in samples])
        b = torch.stack([target_base(infos[d], target_names) for d, _ in samples])
        return t, b

    unscaled = {part: targets_for(samples) for part, samples in split.items()}
    scaler = TargetScaler().fit(unscaled["train"][0])   # training targets only
    datasets = {part: QoRDataset(split[part], graphs, recipes, scaler.transform(t), b)
                for part, (t, b) in unscaled.items()}
    meta = {"target_names": target_names, "target_mode": mode, "design_info": infos,
            "split_sizes": {p: len(s) for p, s in split.items()}}
    return datasets, scaler, meta
