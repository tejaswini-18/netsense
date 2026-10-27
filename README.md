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

## Credit

Uses the OpenABC-D dataset: Chowdhury, Tan, Karri, Garg, "OpenABC-D: A Large-Scale Dataset
For Machine Learning Guided Integrated Circuit Synthesis", arXiv:2110.11292.
