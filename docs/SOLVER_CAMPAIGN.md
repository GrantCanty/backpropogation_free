# Solver study campaign

[`scripts/run_solver_campaign.py`](../scripts/run_solver_campaign.py) runs the
three follow-up studies without changing the solver implementations or the
shared evaluator:

1. a 50-seed confirmation on bundled 8x8 digits;
2. a 20-seed transfer to a second dataset;
3. a 20-seed feature-width sweep at widths 16, 32, 64, and 128.

The confirmatory comparison contains full exact RLS, memory-matched smaller
exact RLS, and Frequent-Directions ridge. Hyperparameters are read from an
existing development/held-out result and frozen. The report includes a paired
bootstrap interval for smaller-minus-full final accuracy and evaluates a
predeclared equivalence margin of one percentage point by default.

## Dataset arguments

`--dataset` is required for the cross-dataset and width-scaling stages:

- `digits` uses scikit-learn's bundled dataset and never downloads anything;
- `fashion_mnist` uses the canonical 60,000/10,000 OpenML split and refuses to
  run unless `--allow-download` is present;
- `npz` reads an existing file supplied with `--dataset-path`. It requires the
  arrays `train_images`, `train_labels`, `test_images`, and `test_labels`.

Fashion-MNIST images are 28 by 28. The default cloud campaign uses at most
1,000 training events per stream segment and 100 locked-evaluation examples
per class. These values are arguments so larger budgets remain explicit.

## Commands

Run the 50-seed local confirmation without network access:

```bash
PYTHONPATH=src python3 scripts/run_solver_campaign.py \
  --stage confirmatory \
  --confirm-seeds 50 \
  --output results/solver_campaign
```

Run the cross-dataset study on a cloud CPU and explicitly permit the download:

```bash
PYTHONPATH=src python3 scripts/run_solver_campaign.py \
  --stage cross-dataset \
  --dataset fashion_mnist \
  --allow-download \
  --dataset-cache .dataset-cache \
  --cross-seeds 20 \
  --jobs 8 \
  --output results/solver_campaign
```

Run width scaling on the same cloud dataset:

```bash
PYTHONPATH=src python3 scripts/run_solver_campaign.py \
  --stage width-scaling \
  --dataset fashion_mnist \
  --allow-download \
  --dataset-cache .dataset-cache \
  --width-seeds 20 \
  --widths 16,32,64,128 \
  --jobs 8 \
  --output results/solver_campaign
```

Use `--stage all` to execute all three in sequence. `--jobs` parallelizes
independent seeds with processes. Use `--jobs 1` for uncontended latency
measurements; multi-process timing is labelled as contention-affected and is
primarily suitable for reducing wall-clock time.

Each stage writes its aggregate JSON, Markdown report, and individual raw
seed files. The campaign root contains `campaign.json` and `CAMPAIGN.md`.
