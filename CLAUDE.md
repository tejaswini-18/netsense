# CLAUDE.md

## Project
NetSense-QoR: a GNN-based logic synthesis QoR predictor. It predicts post-synthesis area and delay of a circuit from its And-Inverter Graph (AIG) plus a 20-step ABC synthesis recipe, using the OpenABC-D dataset. Solo university deep learning project.

The full plan, architecture, experiments and milestones are in PRD.md. Read it when starting a new milestone or when unsure what to build next. Data format findings are in docs/data_notes.md; trust that over assumptions.

## Current milestone
M5 done (baseline results in docs/findings.md). Next: M6 experiments, not started. Dataset is at OPENABC2_DATASET/ (read-only, gitignored).

## About me
Undergraduate (CSE, AI/ML). Comfortable with Python and basic ML. New to PyTorch, graph neural networks and logic synthesis. I need to understand everything for a viva.

## Rules
- Explain what each file or block of code does and why, in plain language. Flag concepts I should know for the viva.
- Work in small steps. Test each piece before the next. Ask before starting a new milestone.
- Prefer simple, readable code with short comments explaining intent.
- Never delete, move or modify anything in data/. It is read-only.
- Ask before installing a library; add every installed library to requirements.txt.
- Never hardcode data paths; take them from config or command-line arguments. Use pathlib.
- If the dataset format is unclear, inspect the files and report what you find. Do not guess.
- Port old OpenABC code to current library versions rather than downgrading.
- When something fails, explain the cause before fixing it.

## Git and secrets
- Do not add "Co-Authored-By: Claude" or "Generated with Claude Code" lines to commit messages or PR descriptions.
- Write short, clear commit messages describing what changed.
- Never read, print, or commit .env files or any API keys or tokens. Use .env.example with placeholder values instead.
- Never commit data/, runs/, checkpoints, or anything listed in .gitignore. Run git status before committing and tell me what will be committed.
- If a new kind of file should be ignored (outputs, caches, checkpoints), add it to .gitignore and tell me.

## Commands
- Run tests: pytest
- Train: python -m src.train --config configs/<name>.yaml
