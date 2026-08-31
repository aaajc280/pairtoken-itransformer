# PairToken iTransformer

Source code for the PairToken iTransformer experiments accompanying a Master
of Finance thesis. The repository contains the model, causal feature
construction, walk-forward training, portfolio evaluation, and the separate
confirmation workflow.

Market data, prepared arrays, model checkpoints, forecasts, results, logs,
runtime captures, and the LaTeX thesis are intentionally excluded. Repository
paths were normalized for publication; the model architecture, transformations,
training settings, random seeds, portfolio rules, and evaluation logic remain
unchanged.

## Model

At each hourly decision, all 276 oriented pairs from a fixed 24-contract
universe are represented as tokens. Each token contains:

- 60 coordinates from the previous completed hour's one-minute pair path;
- 168 completed hourly pair endpoints;
- two causal hedge-weight coordinates; and
- two UTC time coordinates.

A shared projection maps each 232-dimensional token to 48 dimensions. A
single Transformer encoder layer with four attention heads operates across the
pair-token axis, and a shared output head forecasts next-hour fixed-quantity
pair cashflow. Training uses masked Smooth-L1 loss, AdamW, three predetermined
seeds, and arithmetic averaging of their forecasts.

The native representation (`N02`) and causal control (`C02`) use identical
code and hyperparameters. They differ only in the first 60 coordinates: `N02`
retains the chronological one-minute increments, whereas `C02` replaces the
increments within each consecutive 15-minute block by that block's mean.

## Repository layout

- `src/pairtoken/model/`: iTransformer, scaling, loss, seeding, and pair ranking
- `src/pairtoken/development/`: feature construction, training, and evaluation
- `src/pairtoken/history/`: retrospective walk-forward workflow
- `src/pairtoken/confirmation/`: held-out preparation, training, and evaluation
- `src/pairtoken/contracts/`: fixed universe and experiment configuration
- `dependencies/pairs_research/`: market-data and portfolio-ledger modules

## Tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
```

The default test target excludes archive-integration checks because no market
data are distributed with this repository. Use `make test-integration` after
supplying the required public Binance USD-M Futures archives and acquisition
manifests.

This is research code, not a live-trading system. It makes no claim about
future performance or execution feasibility. No software licence is granted;
all rights are reserved unless the repository owner adds one explicitly.
