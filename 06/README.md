# Lab 6 — Training pipelines and project proposal

[Report and measured results](report.md) · [Individual project proposal](proposal.md)

## Run

From this directory with Python 3.11:

```sh
python -m pip install -r requirements.txt
python study.py
```

The submission ZIP includes Fashion-MNIST in `data/`. A checkout downloads the data on first run. To use another cache, set `data_dir` in `config.yaml` to the directory containing `FashionMNIST/`. Relative data and run directories resolve from this directory. The profiling, worker and batch-size runs execute sequentially. The learning-curve runs can overlap. Set `study.accuracy_jobs` to 1 to run them sequentially. To reproduce into an independent directory, give a new `output_dir` in the YAML before rerunning. `python study.py --collect-only` regenerates the tables, plots, and report from completed runs. To continue a matrix that stopped between checkpoints, pass `python study.py --continue-existing`. Use `--evidence PATH` to select a different evidence folder.

The single-run entry point and checkpoint resume can also be exercised directly:

```sh
python train.py --config config.yaml --name manual --stop-after 3
python train.py --config config.yaml --name manual --resume runs/manual/last.pt
```

Here `--stop-after 3` simulates interruption at the epoch boundary. The same YAML retains the full training budget throughout. `train.py --profile` captures an annotated Chrome trace for epoch 1. Full runs save epoch histories, the last checkpoint, and the MLflow tracker under `runs/`.

## Inspect the MLflow runs

The [run inventory](results/mlflow_runs.json) lists all 19 runs. The [per-epoch results](results/epochs.csv) include metrics for every run. After reproducing the study, open the local tracker:

```sh
mlflow ui --backend-store-uri ./runs/mlruns
```

The submission ZIP includes the original MLflow snapshot, full profiler trace, 19 final checkpoints, and the epoch-3 interruption checkpoint. Run `python restore_mlflow.py` from the extracted submission ZIP to restore the original tracker.
