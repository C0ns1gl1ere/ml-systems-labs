"""Fashion-MNIST compression experiment. Run python study.py --help."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST


TARGETS = (0, .25, .5, .75, .9, .95)
FILTER_TARGETS = (.10, .25, .50)
BASE = (16, 32, 3328)
STUDENT = (8, 16, 640)


class FashionData(FashionMNIST):
    mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        *FashionMNIST.mirrors,
    ]


class CNN(nn.Module):
    def __init__(self, widths):
        super().__init__()
        first, second, hidden = widths
        self.widths = widths
        self.conv1 = nn.Conv2d(1, first, 3, padding=1)
        self.conv2 = nn.Conv2d(first, second, 3, padding=1)
        self.fc1 = nn.Linear(second * 9, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.max_pool2d(self.conv1(x).relu(), 2)
        x = F.max_pool2d(self.conv2(x).relu(), 2)
        return self.fc2(self.fc1(x.flatten(1)).relu())


def load_data(root, device):
    datasets = []
    for train in (True, False):
        prefix = "train" if train else "t10k"
        raw = root / "FashionMNIST" / "raw"
        image_path = raw / f"{prefix}-images-idx3-ubyte"
        label_path = raw / f"{prefix}-labels-idx1-ubyte"
        if image_path.is_file() and label_path.is_file():
            images = np.fromfile(image_path, dtype=np.uint8, offset=16)
            labels = np.fromfile(label_path, dtype=np.uint8, offset=8)
            x = torch.from_numpy(images.reshape(-1, 28, 28))
            y = torch.from_numpy(labels).long()
        else:
            data = FashionData(root, train=train, download=True)
            x, y = data.data, data.targets
        x = (x.float().unsqueeze(1) / 255 - .2860) / .3530
        datasets.append((x.to(device), y.to(device)))
    return datasets


@torch.inference_mode()
def logits(model, x, batch_size):
    model.eval()
    return torch.cat([model(batch) for batch in x.split(batch_size)])


def tune(model, train, epochs, batch_size, seed, masks=None, teacher_logits=None, lr=.001):
    x, y = train
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if masks is None:
        masks = {}
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(y), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[indices])
            loss = F.cross_entropy(pred, y[indices])
            if teacher_logits is not None:
                temperature, alpha = 4, .5
                log_student = F.log_softmax(pred / temperature, dim=1)
                log_teacher = F.log_softmax(teacher_logits[indices] / temperature, dim=1)
                # PyTorch takes log Q first for KL(P || Q).
                kl = F.kl_div(log_teacher, log_student, log_target=True, reduction="batchmean")
                loss = alpha * loss + (1 - alpha) * temperature**2 * kl
            loss.backward()
            for parameter, mask in masks.items():
                parameter.grad.mul_(mask)
            optimizer.step()
            with torch.no_grad():
                for parameter, mask in masks.items():
                    parameter.mul_(mask)
    if masks:
        assert all(torch.count_nonzero(p.detach()[~m]).item() == 0 for p, m in masks.items())


def smallest_mask(values, fraction):
    mask = torch.ones_like(values, dtype=torch.bool)
    count = round(values.numel() * fraction)
    if count:
        indices = torch.argsort(values, stable=True)[:count]
        mask[indices] = False
    return mask


def unstructured(baseline, fraction, global_ranking):
    model = copy.deepcopy(baseline)
    weights = [p for p in model.parameters() if p.ndim > 1]
    values = [weight.detach().abs().flatten() for weight in weights]
    if global_ranking:
        masks = smallest_mask(torch.cat(values), fraction).split([v.numel() for v in values])
    else:
        masks = [smallest_mask(v, fraction) for v in values]
    masks = {weight: mask.reshape_as(weight) for weight, mask in zip(weights, masks)}
    with torch.no_grad():
        for weight, mask in masks.items():
            weight.mul_(mask)
    return model, masks


@torch.no_grad()
def structured(baseline, fraction):
    first, second = [
        smallest_mask(conv.weight.flatten(1).norm(p=2, dim=1), fraction)
        .nonzero().flatten()
        for conv in (baseline.conv1, baseline.conv2)
    ]
    model = CNN((len(first), len(second), baseline.widths[-1])).to(first.device)
    model.conv1.weight.copy_(baseline.conv1.weight[first])
    model.conv1.bias.copy_(baseline.conv1.bias[first])
    model.conv2.weight.copy_(baseline.conv2.weight[second][:, first])
    model.conv2.bias.copy_(baseline.conv2.bias[second])
    columns = (second[:, None] * 9 + torch.arange(9, device=second.device)).flatten()
    model.fc1.weight.copy_(baseline.fc1.weight[:, columns])
    model.fc1.bias.copy_(baseline.fc1.bias)
    model.fc2.load_state_dict(baseline.fc2.state_dict())
    return model


def sparse_bytes(model, path):
    # CSR uses 32-bit column indices and row pointers. Biases stay dense.
    state = {}
    for name, parameter in model.state_dict().items():
        if parameter.ndim == 1:
            state[name] = parameter.numpy()
            continue
        array = parameter.numpy().reshape(parameter.size(0), -1)
        rows, columns = np.nonzero(array)
        offsets = np.r_[0, np.bincount(rows, minlength=len(array)).cumsum()]
        state[name] = array.shape
        state[name + "_p"] = offsets.astype(np.int32)
        state[name + "_i"] = columns.astype(np.int32)
        state[name + "_v"] = array[rows, columns]
    # NPZ STORED uses an uncompressed ZIP container, as does torch.save.
    state["widths"] = model.widths
    np.savez(path, **state)
    with np.load(path, allow_pickle=False) as saved:
        for name, parameter in model.state_dict().items():
            if parameter.ndim == 1:
                decoded = saved[name].astype(np.float32)
            else:
                decoded = np.zeros(saved[name], dtype=np.float32)
                counts = np.diff(saved[name + "_p"])
                decoded[np.repeat(np.arange(len(counts)), counts), saved[name + "_i"]] = saved[name + "_v"]
            np.testing.assert_array_equal(decoded.reshape(parameter.shape), parameter.numpy())
    return path.stat().st_size


@torch.inference_mode()
def latency(model, sample):
    model.eval()
    for _ in range(50):
        model(sample)
    rounds = []
    for _ in range(9):
        start = time.perf_counter_ns()
        for _ in range(100):
            model(sample)
        rounds.append((time.perf_counter_ns() - start) / 100_000)
    return statistics.median(rounds), min(rounds), max(rounds)


def metrics(model, test, baseline_count, method, target, train_seconds, batch_size):
    x, y = test
    accuracy = 100 * (logits(model, x, batch_size).argmax(1) == y).float().mean().item()
    cpu = copy.deepcopy(model).cpu().eval()
    weights = [p.detach() for p in cpu.parameters() if p.ndim > 1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        torch.save({"widths": cpu.widths, "weights": cpu.state_dict()}, path / "dense.pt")
        dense_size = (path / "dense.pt").stat().st_size
        sparse_size = sparse_bytes(cpu, path / "csr.npz")
    median, low, high = latency(cpu, x[0:1].cpu())
    multiply_counts = (14 * 14, 7 * 7, 1, 1)
    return {
        "method": method, "target": target, "test_accuracy_pct": round(accuracy, 4),
        "parameters": sum(p.numel() for p in cpu.parameters()),
        "weight_zero_pct": round(100 * (1 - sum(torch.count_nonzero(w).item() for w in weights)
                                         / sum(w.numel() for w in weights)), 4),
        "weight_reduction_pct": round(100 * (1 - sum(w.numel() for w in weights) / baseline_count), 4),
        "dense_bytes": dense_size, "csr_bytes": sparse_size,
        "cpu_median_us": round(median, 3), "cpu_min_block_us": round(low, 3),
        "cpu_max_block_us": round(high, 3),
        "dense_macs": sum(w.numel() * m for w, m in zip(weights, multiply_counts)),
        "nonzero_macs": sum(torch.count_nonzero(w).item() * m for w, m in zip(weights, multiply_counts)),
        "training_seconds": round(train_seconds, 2),
        "channels": f"{cpu.widths[0]}/{cpu.widths[1]}",
    }


def plot(rows, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for method, axis, field, name in [
        ("global", left, "weight_zero_pct", "Global ranking"),
        ("layerwise", left, "weight_zero_pct", "Layer-wise ranking"),
        ("structured", right, "weight_reduction_pct", "Filter removal"),
    ]:
        points = [row for row in rows if row["method"] == method]
        if method == "structured":
            points.insert(0, rows[0])
        axis.plot([row[field] for row in points], [row["test_accuracy_pct"] for row in points],
                  marker="o", label=name)
        if method == "structured":
            for row in points[1:]:
                axis.annotate(f'{100 * row["target"]:.0f}% filters',
                              (row[field], row["test_accuracy_pct"]), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=8)
    left.set(xlabel="Measured zero weights (%)", ylabel="Test accuracy (%)", title="Unstructured")
    right.set(xlabel="Weight elements removed (%)", title="Structured")
    for axis in (left, right):
        axis.legend()
        axis.grid(alpha=.25)
    fig.savefig(path / "sparsity.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, help="Restrict training data for a smoke test")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    train, test = load_data(args.data, device)
    if args.limit:
        indices = torch.randperm(len(train[1]))[:args.limit]
        train = tuple(t[indices] for t in train)
    rows = []
    baseline = CNN(BASE).to(device)
    baseline_count = sum(p.numel() for p in baseline.parameters() if p.ndim > 1)

    def run(model, method, target, epochs, seed, masks=None, teacher=None, lr=.001):
        print(f"Training {method} {target}", flush=True)
        start = time.perf_counter()
        tune(model, train, epochs, args.batch_size, seed, masks, teacher, lr)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = metrics(model, test, baseline_count, method, target, seconds, args.batch_size)
        rows.append(row)
        print(json.dumps(row), flush=True)
        with (args.out / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    (args.out / "environment.json").write_text(json.dumps({
        "platform": platform.platform(), "cpu": platform.processor(), "threads": args.threads,
        "logical_cores": os.cpu_count(), "training_device": args.device, "pytorch": torch.__version__,
        "python": platform.python_version(), "train_examples": len(train[1]), "test_examples": len(test[1]),
        "seed": 42, "epochs": args.epochs, "fine_tuning_epochs": 3, "batch_size": args.batch_size,
    }, indent=2) + "\n", encoding="utf-8")
    run(baseline, "baseline", 0, args.epochs, 42)
    small = CNN(STUDENT).to(device)
    teacher = logits(baseline, train[0], args.batch_size)
    run(copy.deepcopy(small), "student_plain", 0, args.epochs, 43)
    run(small, "student_distilled", 0, args.epochs, 43, teacher=teacher)
    for method in ("global", "layerwise"):
        for fraction in TARGETS:
            model, masks = unstructured(baseline, fraction, method == "global")
            run(model, method, fraction, 3, 44, masks=masks, lr=.0003)
    for fraction in FILTER_TARGETS:
        run(structured(baseline, fraction), "structured", fraction, 3, 44, lr=.0003)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
