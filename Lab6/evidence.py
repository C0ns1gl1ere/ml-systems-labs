"""Build the report from measured histories and verify MLflow and resume evidence."""
import csv
from importlib.metadata import version
import json
import platform
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow import MlflowClient

from train import PHASES, load_checkpoint, write_csv


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def worker_plot(rows, out):
    workers = [r["workers"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(workers, [r["cold_samples_s"] for r in rows], "o--", label="Epoch 1 (startup)")
    ax.plot(workers, [r["steady_samples_s"] for r in rows], "s-", label="Epoch 2 (warm)")
    ax.set(xlabel="DataLoader workers", ylabel="Training samples/s", xticks=workers)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "workers.png", dpi=130)
    plt.close(fig)


def curve_plot(runs, series, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    metrics = (("train_loss", "Mean training loss"), ("val_accuracy", "Validation accuracy"))
    for axis, (key, label) in zip(axes, metrics):
        for name, rows, style in runs:
            axis.plot([r["epoch"] for r in rows], [r[key] for r in rows], style, label=name)
        if series is not None:
            axis.axvline(series + .5, color="0.45", linestyle=":", label="Checkpoint")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def export_profile_trace(output, out):
    """Keep annotated training regions in a compact Chrome/Perfetto trace."""
    import gzip

    trace = json.loads((output / "profile" / "profile.json").read_text(encoding="utf-8"))
    events = []
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("name") in PHASES:
            events.append({key: event[key] for key in ("name", "ph", "ts", "dur", "pid", "tid")})
    assert {event["name"] for event in events} == set(PHASES)
    compressed = gzip.compress(json.dumps({"traceEvents": events}).encode("utf-8"), mtime=0)
    (out / "profile_spans.json.gz").write_bytes(compressed)

def export_tracking(output, out, inventory):
    """Package original run records and small artifacts without local paths."""
    from zipfile import ZIP_DEFLATED, ZipFile

    import yaml

    tracking = output / "mlruns"
    with ZipFile(out / "mlflow_snapshot.zip", "w", compression=ZIP_DEFLATED) as archive:
        for record in inventory:
            run_id = record["run_id"]
            meta, = tracking.glob(f"*/{run_id}/meta.yaml")
            run = meta.parent
            experiment = run.parent / "meta.yaml"
            exp_key = "mlruns/" + experiment.relative_to(tracking).as_posix()
            if exp_key not in archive.namelist():
                data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
                data["artifact_location"] = "file:///artifacts"
                archive.writestr(exp_key, yaml.safe_dump(data))
            data = yaml.safe_load(meta.read_text(encoding="utf-8"))
            data["artifact_uri"] = f"file:///artifacts/{run_id}"
            data["user_id"] = "lab6"
            archive.writestr("mlruns/" + meta.relative_to(tracking).as_posix(),
                             yaml.safe_dump(data))
            for group in ("metrics", "params", "tags"):
                for path in sorted((run / group).iterdir()):
                    if path.name == "mlflow.user":
                        continue
                    key = "mlruns/" + path.relative_to(tracking).as_posix()
                    if path.name == "mlflow.source.name":
                        archive.writestr(key, "Lab6/train.py")
                    elif path.is_file():
                        archive.write(path, key)
            artifacts = output / "artifacts" / run_id / "artifacts"
            for name in ("resolved.yaml", "history.csv", "profile.csv"):
                path = artifacts / name
                if path.is_file():
                    archive.write(path, f"artifacts/{run_id}/{name}")

def collect(cfg, output, out):
    from study import jobs

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    study = cfg["study"]
    names = dict.fromkeys(name for name, _, _ in jobs(cfg))
    runs = {name: read_rows(output / name / "history.csv") for name in names}
    write_csv(out / "epochs.csv", [
        dict(run=name, **row) for name, rows in runs.items() for row in rows
    ])

    profile = list(csv.DictReader((output / "profile" / "profile.csv").read_text().splitlines()))
    write_csv(out / "profile.csv", profile)
    export_profile_trace(output, out)
    phases = {r["phase"]: float(r["share_pct"]) for r in profile}

    workers = []
    for count in study["workers"]:
        cold, warm = runs[f"workers_{count}"]
        workers.append(dict(workers=count, cold_samples_s=cold["samples_per_s"],
                            steady_samples_s=warm["samples_per_s"], peak_rss_mib=max(
                                cold["peak_rss_mib"], warm["peak_rss_mib"])))
    write_csv(out / "workers.csv", workers)
    worker_plot(workers, out)

    batches = []
    for count in study["batches"]:
        rows = runs[f"batch_{count}"]
        batches.append(dict(
            batch=count, samples_s=sum(r["samples"] for r in rows) / sum(r["seconds"] for r in rows),
            peak_rss_mib=max(r["peak_rss_mib"] for r in rows),
            peak_cuda_mib=max(r["peak_cuda_mib"] for r in rows),
            final_accuracy=rows[-1]["val_accuracy"],
        ))
    write_csv(out / "batches.csv", batches)

    comparisons = []
    for optimizer in ("sgd", "adam"):
        for scheduler in ("step", "cosine", "plateau"):
            rows = runs[f"{optimizer}_{scheduler}"]
            comparisons.append(dict(
                optimizer=optimizer, scheduler=scheduler, initial_lr=rows[0]["lr"],
                final_epoch_lr=rows[-1]["lr"], final_accuracy=rows[-1]["val_accuracy"],
                best_accuracy=max(row["val_accuracy"] for row in rows),
            ))
    write_csv(out / "optimizers.csv", comparisons)

    reference, resumed = runs["adam_cosine"], runs["resumed"]
    state_a = load_checkpoint(output / "adam_cosine" / "last.pt")
    state_b = load_checkpoint(output / "resumed" / "last.pt")
    cut = study["interrupt_after"]
    assert [int(r["epoch"]) for r in resumed] == list(range(1, len(reference) + 1))
    assert len(reference) == study["comparison_epochs"]
    weight_delta = max((p - state_b["model"][key]).abs().max().item()
                       for key, p in state_a["model"].items())
    resume_deltas = {key: max(abs(a[key] - b[key]) for a, b in zip(reference, resumed))
                     for key in ("train_loss", "val_loss", "val_accuracy", "lr")}
    resume_check = dict(
        interruption_epoch=cut, max_weight_difference=weight_delta, max_metric_delta=resume_deltas,
        bitwise_final_weights=weight_delta == 0,
        passed=weight_delta <= 1e-6 and max(resume_deltas.values()) <= 1e-6,
    )
    (out / "resume_check.json").write_text(json.dumps(resume_check, indent=2) + "\n")

    accum_a, accum_b = runs["batch_256"], runs["acc_64x4"]
    acc_loss = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(accum_a, accum_b))
    acc_accuracy = abs(accum_a[-1]["val_accuracy"] - accum_b[-1]["val_accuracy"])
    accum_check = dict(
        max_epoch_train_loss_delta=acc_loss, final_accuracy_delta=acc_accuracy,
        similar=acc_loss <= .02 and acc_accuracy <= .01,
    )
    (out / "accumulation_check.json").write_text(json.dumps(accum_check, indent=2) + "\n")
    curve_plot([("Batch 256", accum_a, "o-"), ("Batch 64 × 4", accum_b, "x--")],
               None, out / "accumulation.png")
    curve_plot([("Uninterrupted", reference, "o-"), ("Resumed", resumed, "x--")],
               cut, out / "resume.png")

    client = MlflowClient(tracking_uri=(output / "mlruns").as_uri())
    inventory = []
    for name, rows in runs.items():
        state = load_checkpoint(output / name / "last.pt")
        run_id = state["run_id"]
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert int(run.data.tags["last_completed_epoch"]) == len(rows)
        assert len(rows) == state["config"]["epochs"]
        steps = [int(row["epoch"]) for row in rows]
        for key in rows[0].keys() - {"epoch"}:
            history = sorted(client.get_metric_history(run_id, key), key=lambda item: item.step)
            assert [item.step for item in history] == steps
            assert all(item.value == row[key] for item, row in zip(history, rows))

        root_artifacts = {artifact.path for artifact in client.list_artifacts(run_id)}
        assert {"resolved.yaml", "history.csv", "final"} <= root_artifacts
        assert {"final/last.pt"} == {artifact.path for artifact in client.list_artifacts(run_id, "final")}
        if name == "resumed":
            assert {"interrupted/last.pt"} == {
                artifact.path for artifact in client.list_artifacts(run_id, "interrupted")
            }
        if name == "profile":
            assert {"profile.csv", "profile.json"} <= root_artifacts
        inventory.append(dict(name=name, run_id=run_id, status=run.info.status,
                              metrics_verified=sorted(rows[0].keys() - {"epoch"}),
                              epochs=len(rows), params=run.data.params, artifacts=sorted(root_artifacts)))
    (out / "mlflow_runs.json").write_text(json.dumps(inventory, indent=2) + "\n")

    export_tracking(output, out, inventory)

    versions = {name: version(name) for name in
                ("torch", "torchvision", "mlflow", "numpy", "matplotlib", "psutil", "PyYAML")}
    env = dict(platform=platform.platform(), cpu=platform.processor(),
               logical_cores=os.cpu_count(), python=platform.python_version(),
               device=inventory[0]["params"]["resolved_device"], threads=cfg["threads"],
               packages=versions)
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    leader = max(comparisons, key=lambda r: r["final_accuracy"])
    fastest = max(batches, key=lambda r: r["samples_s"])
    highest = max(batches, key=lambda r: r["final_accuracy"])
    worker_winner = max(workers, key=lambda r: r["steady_samples_s"])
    zero = next(row for row in workers if row["workers"] == 0)
    cold_loser = min(workers, key=lambda r: r["cold_samples_s"])
    warm_low = min(r["steady_samples_s"] for r in workers)
    warm_spread = 100 * (1 - warm_low / worker_winner["steady_samples_s"])
    bottleneck = max(phases, key=phases.get)
    optimizer_text = {"step": "StepLR", "cosine": "CosineAnnealingLR",
                      "plateau": "ReduceLROnPlateau"}
    batch_headers = ["Batch", "Samples/s", "Peak tree RSS (MiB)", "Peak CUDA (MiB)", "Epoch 5 val. acc."]
    profile_table = table(["Phase", "Milliseconds", "Share"], [
        (r["phase"], f"{float(r['milliseconds']):,.1f}", f"{float(r['share_pct']):.1f}%")
        for r in profile
    ])
    worker_table = table(["Workers", "Cold samples/s", "Warm samples/s", "Peak tree RSS (MiB)"], [
        (r["workers"], f"{r['cold_samples_s']:,.0f}", f"{r['steady_samples_s']:,.0f}",
         f"{r['peak_rss_mib']:.1f}") for r in workers
    ])
    batch_table = table(batch_headers, [
        (r["batch"], f"{r['samples_s']:,.0f}", f"{r['peak_rss_mib']:.1f}",
         f"{r['peak_cuda_mib']:.1f}" if env["device"].startswith("cuda") else "—",
         f"{r['final_accuracy']:.2%}") for r in batches
    ])
    optimizer_table = table(["Optimizer", "Schedule", "Start LR", "Epoch 10 LR",
                             "Epoch 10 val. acc.", "Best val. acc."], [
        (r["optimizer"].upper(), optimizer_text[r["scheduler"]], f"{r['initial_lr']:.5g}",
         f"{r['final_epoch_lr']:.5g}", f"{r['final_accuracy']:.2%}",
         f"{r['best_accuracy']:.2%}") for r in comparisons
    ])
    report = f"""# Lab 6 — Training pipelines and project proposal

## 1. Configuration and tracking

The CNN selected in [Lab 3](https://github.com/C0ns1gl1ere/ml-systems-labs/blob/main/lab03/report.md) had 20,490 parameters and 89.64% held-out test accuracy. This lab trains the same architecture from a seeded initialization on the same 55,000/5,000 Fashion-MNIST split, with the same per-image pixel normalization. Comparisons use validation accuracy. The test split is reserved for final model evaluation outside these sweeps. All runs use channels-last PyTorch tensors and the same sample order per epoch. Each run takes one YAML config and saves an atomic epoch checkpoint. The MLflow snapshot in the submission ZIP contains {len(inventory)} completed runs with parameters, epoch metrics, and artifacts. The [run inventory](results/mlflow_runs.json) records run IDs and verified metric steps. [Environment](results/environment.json) records the {env['device'].upper()} device, {cfg['threads']} PyTorch threads, OS and exact package versions.

## 2. One-epoch PyTorch profile

{profile_table}

Shares use the sum of the four inclusive profiler regions. Data loading includes batch fetching and input transfer. Forward includes the loss calculation. Backward computes summed gradients. The optimizer region normalizes gradients by sample count, updates weights and clears gradients. Profiling is separate from throughput measurements. The annotated phase trace in the submission ZIP preserves start times and durations for the four measured regions. The largest measured region was **{bottleneck.replace('_', ' ')}** ({phases[bottleneck]:.1f}%).

## 3. DataLoader worker sweep

{worker_table}

![Worker count and throughput](results/workers.png)

Epoch 1 includes worker startup. Epoch 2 reuses persistent workers. The fastest warm setting was **{worker_winner['workers']} workers**. All warm results were within {warm_spread:.1f}% of the fastest. The slowest cold result used {cold_loser['workers']} workers at {cold_loser['cold_samples_s']:,.0f} samples/s, versus {zero['cold_samples_s']:,.0f} with zero workers. Batch loading accounted for {phases['data_loading']:.1f}% of profiled time and forward/backward for {phases['forward'] + phases['backward']:.1f}%. The main steady-state bottleneck on this machine was model computation, and spawning worker processes added a substantial first-epoch cost.

## 4. Batch-size study

{batch_table}

Throughput is total training samples divided by total elapsed training-epoch time. Evaluation, checkpoint writes and tracking calls are outside the timing. Peak memory is the maximum sampled sum of process-tree RSS at 25 ms intervals during training, including shared pages in each process that maps them. CUDA reports the maximum allocated tensor memory when used. Batch **{fastest['batch']}** gave the highest throughput ({fastest['samples_s']:,.0f} samples/s) with {fastest['peak_rss_mib']:.1f} MiB sampled RSS. Batch **{highest['batch']}** gave the highest epoch-five accuracy ({highest['final_accuracy']:.2%}) with {highest['peak_rss_mib']:.1f} MiB sampled RSS. Smaller batches take more optimizer steps per epoch. A fixed epoch budget therefore measures the combined effect of batch size and update count. The per-epoch timings show variation on this Windows host, so the throughput ranking applies to these five-epoch runs.

## 5. Optimizer and scheduler comparison

{optimizer_table}

Each combination trained for ten epochs. These learning-behavior runs can run concurrently. Their timings are outside the isolated throughput comparison. SGD uses momentum {cfg['momentum']}. The initial learning rate is fixed across schedules within each optimizer. StepLR multiplies it by {cfg['gamma']} every {cfg['step_size']} epochs. Cosine anneals over all ten epochs to {cfg['min_lr_ratio']:.0%} of the initial rate. ReduceLROnPlateau monitors validation loss with patience {cfg['plateau_patience']} and factor {cfg['gamma']}. Epoch 10 LR is the rate used for the final optimization pass before the scheduler updates. **{leader['optimizer'].upper()} with {optimizer_text[leader['scheduler']]}** had the highest final validation accuracy ({leader['final_accuracy']:.2%}) for these starting rates and this seed.

## 6. Accumulation

![Training loss and validation accuracy for true and accumulated batches](results/accumulation.png)

A true batch of 256 and four microbatches of 64 see the same samples before each update. Summed gradients are divided by the actual number of examples in each window, including the final partial window. The largest epoch training-loss gap was {acc_loss:.6g}. The final validation-accuracy gap was {100 * acc_accuracy:.4f} percentage points. The true-batch curve is reused from the identical batch-size run. The [similarity check](results/accumulation_check.json) uses limits of 0.02 for loss and one percentage point for accuracy.

## 7. Checkpoint and resume

![Uninterrupted and resumed learning curves](results/resume.png)

The uninterrupted reference is the identical Adam + cosine run from the scheduler study. The interrupted process exited after saving epoch {cut}. A fresh process restored weights, optimizer, scheduler, RNG states, epoch history and the original MLflow run ID. It continued at epoch {cut + 1} through epoch {len(reference)}. The final maximum absolute parameter gap was {weight_delta:.6g} and the maximum learning-rate or metric gap was {max(resume_deltas.values()):.6g}. The [resume check](results/resume_check.json) records exact weight equality and a 1e-6 tolerance check.

## 8. Individual proposal

[CPU deployment of a quantized clothing classifier](proposal.md). Instructor approval is pending.

## Reading

Vijay Janapa Reddi, *Machine Learning Systems*. The assigned training chapter frames the phase breakdown and trade-offs in worker prefetching and accumulation. The model development and MLOps readings motivate fixed validation data, logged run configurations, and repeatable checkpoint recovery. [Textbook](https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.pdf)
"""
    (out.parent / "report.md").write_text(report, encoding="utf-8")
