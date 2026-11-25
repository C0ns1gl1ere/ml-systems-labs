"""Run the Lab 6 experiment matrix and aggregate measured evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import subprocess
import sys

from train import LAB, load_checkpoint, read_config, resolve


def performance_jobs(cfg):
    study = cfg["study"]
    yield "profile", dict(epochs=1), ["--profile"]
    for workers in study["workers"]:
        yield f"workers_{workers}", dict(num_workers=workers, epochs=2), []
    for batch in study["batches"]:
        yield f"batch_{batch}", dict(batch_size=batch, epochs=study["batch_epochs"]), []


def accuracy_chains(cfg):
    study = cfg["study"]
    for optimizer, scheduler in product(("sgd", "adam"), ("step", "cosine", "plateau")):
        changes = dict(
            optimizer=optimizer, scheduler=scheduler, epochs=study["comparison_epochs"],
            learning_rate=study["optimizer_lr"][optimizer],
        )
        yield [(f"{optimizer}_{scheduler}", changes, [])]
    yield [("acc_64x4", dict(batch_size=64, accumulation=4,
                           epochs=study["batch_epochs"]), [])]
    changes = dict(optimizer="adam", scheduler="cosine", epochs=study["comparison_epochs"],
                   learning_rate=study["optimizer_lr"]["adam"])
    cut = study["interrupt_after"]
    yield [("resumed", changes, ["--stop-after", str(cut)]),
           ("resumed", changes, ["--resume"])]


def jobs(cfg):
    yield from performance_jobs(cfg)
    for chain in accuracy_chains(cfg):
        yield from chain


def run_job(cfg, output, job, continue_existing):
    import yaml

    name, changes, flags = job
    resolved = {**cfg, **changes}
    job_cfg = output / "configs" / (name + ".yaml")
    job_cfg.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    checkpoint = output / name / "last.pt"
    target = int(flags[1]) if flags and flags[0] == "--stop-after" else resolved["epochs"]
    args = [sys.executable, str(LAB / "train.py"), "--config", str(job_cfg), "--name", name]
    if flags == ["--resume"]:
        flags = ["--resume", str(checkpoint)]
    elif continue_existing and checkpoint.is_file():
        saved = load_checkpoint(checkpoint)
        if saved["config"] != {key: value for key, value in resolved.items() if key != "study"}:
            raise ValueError(f"Existing checkpoint for {name} has a different configuration")
        if saved["epoch"] >= target:
            print(f"{name}: reusing {saved['epoch']} completed epoch(s)", flush=True)
            return
        flags = [*flags, "--resume", str(checkpoint)]
    if continue_existing and flags[:1] == ["--resume"]:
        saved = load_checkpoint(checkpoint)
        if saved["epoch"] >= target:
            print(f"{name}: reusing {saved['epoch']} completed epoch(s)", flush=True)
            return
    subprocess.run([*args, *flags], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB / "config.yaml")
    parser.add_argument("--evidence", type=Path, default=LAB / "results")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--continue-existing", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    output = resolve(cfg["output_dir"])
    if not args.collect_only:
        (output / "configs").mkdir(parents=True, exist_ok=True)
        for job in performance_jobs(cfg):
            run_job(cfg, output, job, args.continue_existing)

        def run_chain(chain):
            for job in chain:
                run_job(cfg, output, job, args.continue_existing)

        with ThreadPoolExecutor(max_workers=cfg["study"]["accuracy_jobs"]) as pool:
            for _ in pool.map(run_chain, accuracy_chains(cfg)):
                pass

    from evidence import collect

    collect(cfg, output, args.evidence)


if __name__ == "__main__":
    main()
