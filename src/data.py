"""Dataset class, target normalisation and train/val/test splits (built in M3).

For now: a loader for OpenABC-D graph files, which were saved with PyTorch Geometric 1.7.
Current PyG (2.x) cannot unpickle those objects directly, so we rebuild them (see docs/data_notes.md).
"""

import io
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch
from torch_geometric.data import Data


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
