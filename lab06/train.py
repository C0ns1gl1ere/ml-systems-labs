"""Reproducible Fashion-MNIST training with the winning Lab 3 CNN."""
import argparse
from contextlib import nullcontext
import csv
import os
from pathlib import Path
import random
import threading
import time

import psutil
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler, random_split


LAB = Path(__file__).resolve().parent
PHASES = ("data_loading", "forward", "backward", "optimizer_step")


class FashionTensors(Dataset):
    """Use the same bytes and normalization as Lab 3 without a PIL round trip."""

    def __init__(self, dataset, mean, std):
        self.images = dataset.data
        self.targets = dataset.targets
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = self.images[index].unsqueeze(-3).float().div_(255)
        return pixels.sub_(self.mean).div_(self.std), self.targets[index]


class MemoryMonitor:
    """Sample aggregate process-tree RSS, including data-loader workers."""

    def __init__(self):
        self.peak = 0
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def sample(self):
        rss = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak = max(self.peak, rss)

    def poll(self):
        while not self.stop.wait(0.025):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()


def read_config(path):
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path):
    return (LAB / path).resolve()


def load_data(config):
    from torchvision.datasets import FashionMNIST

    FashionMNIST.mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/fashion-mnist/",
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        *FashionMNIST.mirrors,
    ]
    raw = FashionMNIST(resolve(config["data_dir"]), train=True, download=True)
    train, validation = random_split(
        FashionTensors(raw, config["mean"], config["std"]),
        [config["train_size"], config["validation_size"]],
        generator=torch.Generator().manual_seed(config["split_seed"]),
    )
    return train, validation


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
    )


def make_optimizer(model, cfg):
    kwargs = {"lr": cfg["learning_rate"]}
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=cfg["momentum"], **kwargs)
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


def make_scheduler(optimizer, cfg):
    from torch.optim import lr_scheduler

    name = cfg["scheduler"]
    if name == "constant":
        return None
    if name == "step":
        return lr_scheduler.StepLR(optimizer, cfg["step_size"], gamma=cfg["gamma"])
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, cfg["epochs"], eta_min=cfg["learning_rate"] * cfg["min_lr_ratio"]
        )
    if name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["gamma"], patience=cfg["plateau_patience"],
            min_lr=cfg["learning_rate"] * cfg["min_lr_ratio"],
        )
    raise ValueError(f"Unknown scheduler: {name}")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, device, accumulation, profile, directory):
    model.train()
    loss_total = torch.zeros((), device=device)
    seen = window_samples = 0
    context = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        if profile else nullcontext()
    )
    region = torch.profiler.record_function if profile else lambda _: nullcontext()
    sync(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with context as trace:
        with region("data_loading"):
            batches = iter(loader)
        for index in range(len(loader)):
            with region("data_loading"):
                images, targets = next(batches)
                images, targets = images.to(device, memory_format=torch.channels_last), targets.to(device)
                if profile:
                    sync(device)
            with region("forward"):
                loss = nn.functional.cross_entropy(model(images), targets, reduction="sum")
                if profile:
                    sync(device)
            with region("backward"):
                loss.backward()
                if profile:
                    sync(device)
            seen += len(targets)
            window_samples += len(targets)
            loss_total += loss.detach()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                with region("optimizer_step"):
                    for parameter in model.parameters():
                        parameter.grad.div_(window_samples)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if profile:
                        sync(device)
                window_samples = 0
    sync(device)
    elapsed = time.perf_counter() - start
    if profile:
        times = {e.key: e.cpu_time_total / 1000 for e in trace.key_averages() if e.key in PHASES}
        total = sum(times.values())
        write_csv(directory / "profile.csv", [
            {"phase": phase, "milliseconds": times[phase], "share_pct": 100 * times[phase] / total}
            for phase in PHASES
        ])
        trace.export_chrome_trace(str(directory / "profile.json"))
    return loss_total.item() / seen, elapsed, seen


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    losses = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    for images, targets in loader:
        images, targets = images.to(device, memory_format=torch.channels_last), targets.to(device)
        logits = model(images)
        losses += nn.functional.cross_entropy(logits, targets, reduction="sum")
        correct += (logits.argmax(dim=1) == targets).sum()
    return losses.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_metrics(directory, run_id, history):
    """Replay missing tracking writes if a process exits after a checkpoint save."""
    import mlflow

    if not history:
        return
    client = mlflow.MlflowClient()
    steps_by_metric = {
        key: {point.step for point in client.get_metric_history(run_id, key)}
        for key in history[0] if key != "epoch"
    }
    for row in history:
        step = row["epoch"]
        missing = {key: row[key] for key, steps in steps_by_metric.items() if step not in steps}
        if missing:
            mlflow.log_metrics(missing, step=step)
    write_csv(directory / "history.csv", history)

def run_training(config, name, *, profile=False, stop_after=None, resume=None):
    import mlflow
    import yaml

    if min(config[k] for k in ("threads", "batch_size", "accumulation", "epochs")) < 1:
        raise ValueError("threads, batch_size, accumulation and epochs must be positive")
    if stop_after is not None and not 1 <= stop_after < config["epochs"]:
        raise ValueError("stop-after must leave at least one epoch to resume")
    cfg = {key: value for key, value in config.items() if key != "study"}
    output = resolve(cfg["output_dir"])
    directory = output / name
    if resume:
        state = load_checkpoint(resume)
        if state["config"] != cfg or state["name"] != name:
            raise ValueError("Resume requires the original configuration and run name")
        if state["epoch"] >= (stop_after or cfg["epochs"]):
            raise ValueError("Checkpoint already reached the requested final epoch")
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    else:
        directory.mkdir(parents=True, exist_ok=False)

    mlflow.set_tracking_uri((output / "mlruns").as_uri())
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=(output / "artifacts").as_uri())
    mlflow.set_experiment(cfg["experiment"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    requested = cfg["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    train, validation = load_data(cfg)
    if cfg["num_workers"]:
        # Subset indices stay in Python, and the image data has one shared storage.
        train.dataset.images.share_memory_()
        train.dataset.targets.share_memory_()
    # A dedicated generator prevents worker lifecycle from changing shuffle order.
    shuffle_rng = torch.Generator().manual_seed(cfg["seed"])
    worker_rng = torch.Generator().manual_seed(cfg["seed"] + 1)
    sampler = torch.utils.data.RandomSampler(train, generator=shuffle_rng)
    train_loader = DataLoader(
        train, batch_size=None,
        sampler=BatchSampler(sampler, cfg["batch_size"], drop_last=False),
        num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0,
        generator=worker_rng,
    )
    val_loader = DataLoader(
        validation, batch_size=None, generator=worker_rng,
        sampler=BatchSampler(SequentialSampler(validation), 128, drop_last=False),
    )
    model = cnn().to(device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    history = []
    run_id = None
    if resume:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        shuffle_rng.set_state(state["shuffle_rng"])
        worker_rng.set_state(state["worker_rng"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        history = state["history"]
        run_id = state["run_id"]

    with mlflow.start_run(run_id=run_id, run_name=name) as run:
        if resume:
            restore_metrics(directory, run_id, history)
        else:
            mlflow.log_params({
                **cfg, "resolved_device": str(device), "architecture": "Lab 3 CNN",
                "parameters": sum(p.numel() for p in model.parameters()),
            })
            path = directory / "resolved.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            mlflow.log_artifact(str(path))
        target_epoch = stop_after or cfg["epochs"]
        for epoch in range(len(history) + 1, target_epoch + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr = optimizer.param_groups[0]["lr"]
            with MemoryMonitor() as memory:
                train_loss, seconds, samples = train_epoch(
                    model, train_loader, optimizer, device, cfg["accumulation"],
                    profile and epoch == 1, directory,
                )
                cuda_peak = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else 0.0
                )
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()
            row = dict(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy,
                lr=lr, next_lr=optimizer.param_groups[0]["lr"], seconds=seconds, samples=samples,
                samples_per_s=samples / seconds, peak_rss_mib=memory.peak / 2**20,
                peak_cuda_mib=cuda_peak,
            )
            history.append(row)
            checkpoint = dict(
                epoch=epoch, history=history, name=name, config=cfg, run_id=run.info.run_id,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                shuffle_rng=shuffle_rng.get_state(), worker_rng=worker_rng.get_state(),
            )
            checkpoint_path = directory / "last.pt"
            pending_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, pending_path)
            pending_path.replace(checkpoint_path)
            write_csv(directory / "history.csv", history)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"{name} {epoch}/{target_epoch}: val={val_accuracy:.4f}, {samples / seconds:.0f}/s",
                  flush=True)
        mlflow.log_artifact(str(directory / "history.csv"))
        artifact_name = "interrupted" if stop_after else "final"
        mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_name)
        if profile:
            for path in directory.glob("profile.*"):
                mlflow.log_artifact(str(path))
        mlflow.set_tag("last_completed_epoch", len(history))
        return history, run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB / "config.yaml")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_training(read_config(args.config), args.name, profile=args.profile,
                 stop_after=args.stop_after, resume=args.resume)


if __name__ == "__main__":
    main()
