# NetSense-QoR

A graph neural network that predicts the post-synthesis area and delay of a digital circuit
from its And-Inverter Graph (AIG) and a 20-step ABC synthesis recipe, trained on the
OpenABC-D dataset.

> Work in progress. See `PRD.md` for the full plan.

## How to run

Setup (Windows PowerShell):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Train and evaluate (writes `runs/<experiment>_<timestamp>/` with metrics, checkpoint and plots):

```
python -m src.train --config configs/baseline.yaml --data-root <path to OpenABC-D folder>
```

Training on Kaggle: build a compact copy (44 files, ~80 MB) of only the data the pipeline reads, upload it as a Kaggle
dataset, then follow `notebooks/02_kaggle_train.ipynb`. On Kaggle, install only `torch_geometric` (and `pytest`);
Kaggle's preinstalled PyTorch must not be reinstalled from `requirements.txt`.

```
python -m src.make_subset --data-root <path to OpenABC-D folder> --out kaggle_data/openabc-compact
```

## Credit

Uses the OpenABC-D dataset: Chowdhury, Tan, Karri, Garg, "OpenABC-D: A Large-Scale Dataset
For Machine Learning Guided Integrated Circuit Synthesis", arXiv:2110.11292.
