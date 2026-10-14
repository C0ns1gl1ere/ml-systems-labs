"""Lab 2: two-hidden-layer network, implemented with NumPy only."""
from pathlib import Path
import csv
import json
import platform
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


def prepare_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    Y = np.eye(10, dtype=np.float64)[y]
    return train_test_split(X, Y, y, test_size=0.30,
                            stratify=y, random_state=42)


def initialize(width, seed=42):
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, 0.01, (64, width)), b1=np.zeros(width),
                W2=rng.normal(0, 0.01, (width, width)), b2=np.zeros(width),
                W3=rng.normal(0, 0.01, (width, 10)), b3=np.zeros(10))


def forward(X, p):
    Z1 = X @ p['W1'] + p['b1']
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ p['W2'] + p['b2']
    A2 = np.maximum(0.0, Z2)
    Z3 = A2 @ p['W3'] + p['b3']
    shifted = Z3 - Z3.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    return probs, (X, Z1, A1, Z2, A2, probs, log_probs)


def cross_entropy(Y, cache):
    # Log-softmax avoids log(0), and preserves the exact gradient formula.
    return float(-np.sum(Y * cache[-1]) / len(Y))


def backward(Y, p, cache):
    X, Z1, A1, Z2, A2, probs, _ = cache
    D3 = (probs - Y) / len(Y)
    dW3 = A2.T @ D3
    db3 = D3.sum(axis=0)
    D2 = (D3 @ p['W3'].T) * (Z2 > 0)
    dW2 = A1.T @ D2
    db2 = D2.sum(axis=0)
    D1 = (D2 @ p['W2'].T) * (Z1 > 0)
    return dict(W1=X.T @ D1, b1=D1.sum(axis=0),
                W2=dW2, b2=db2, W3=dW3, b3=db3)


def numerical_gradient_check(X, Y, p, seed=2026, epsilon=1e-5):
    """Ten uniformly sampled weight entries, with no retries/filtering."""
    _, cache = forward(X, p)
    analytic = backward(Y, p, cache)
    names = ['W1', 'W2', 'W3']
    sizes = [p[n].size for n in names]
    chosen = np.random.default_rng(seed).choice(sum(sizes), 10, replace=False)
    rows = []
    for flat_index in chosen:
        offset = int(flat_index)
        for name, size in zip(names, sizes):
            if offset < size:
                break
            offset -= size
        index = np.unravel_index(offset, p[name].shape)
        original = p[name][index]
        try:
            p[name][index] = original + epsilon
            _, plus = forward(X, p)
            lp = cross_entropy(Y, plus)
            p[name][index] = original - epsilon
            _, minus = forward(X, p)
            lm = cross_entropy(Y, minus)
        finally:
            p[name][index] = original
        numerical = (lp - lm) / (2 * epsilon)
        exact = float(analytic[name][index])
        relative = abs(numerical - exact) / max(1e-12, abs(numerical) + abs(exact))
        kink = (np.any((plus[1] > 0) != (minus[1] > 0)) or
                np.any((plus[3] > 0) != (minus[3] > 0)))
        rows.append(dict(parameter=name, row=int(index[0]), column=int(index[1]),
                         analytic=exact, numerical=numerical, relative_error=relative,
                         passed=bool(relative < 1e-6), relu_kink=bool(kink)))
    return rows


def train(X, Y, width, learning_rate, epochs=50, batch_size=32):
    p = initialize(width)
    rng = np.random.default_rng(123)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            _, cache = forward(X[idx], p)
            grads = backward(Y[idx], p, cache)
            for name in p:
                p[name] -= learning_rate * grads[name]
        probs, cache = forward(X, p)
        loss = cross_entropy(Y, cache)
        accuracy = float(np.mean(probs.argmax(axis=1) == Y.argmax(axis=1)))
        history.append(dict(width=width, learning_rate=learning_rate,
                            epoch=epoch, train_loss=loss, train_accuracy=accuracy))
        print(f'W={width:2d} lr={learning_rate:.3f} epoch={epoch:02d} '
              f'loss={loss:.6f} train_acc={accuracy:.4f}', flush=True)
    return p, history


def time_passes(X, Y, p, repeats=100):
    # Both measurements exclude dataset loading, updates, and explicit loss.
    for _ in range(20):
        _, cache = forward(X, p)
        backward(Y, p, cache)
    _, cache = forward(X, p)
    forward_ns, backward_ns = [], []
    for i in range(repeats):
        # Alternate ordering to reduce systematic ordering bias.
        if i % 2 == 0:
            t = time.perf_counter_ns(); forward(X, p)
            forward_ns.append(time.perf_counter_ns() - t)
            t = time.perf_counter_ns(); backward(Y, p, cache)
            backward_ns.append(time.perf_counter_ns() - t)
        else:
            t = time.perf_counter_ns(); backward(Y, p, cache)
            backward_ns.append(time.perf_counter_ns() - t)
            t = time.perf_counter_ns(); forward(X, p)
            forward_ns.append(time.perf_counter_ns() - t)
    f, b = np.median(forward_ns) / 1e6, np.median(backward_ns) / 1e6
    return dict(batch_size=len(X), runs=repeats, forward_ms=float(f),
                backward_ms=float(b), backward_forward_ratio=float(b/f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def run_experiments(output_dir='results'):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    X_train, X_test, Y_train, Y_test, y_train, y_test = prepare_data()
    checks, histories, summaries, timings, sanity = [], [], [], [], []
    models = {}
    with threadpool_limits(limits=1):
        for width in [16, 64]:
            initial = initialize(width)
            probs, cache = forward(X_train[:32], initial)
            assert probs.shape == (32, 10)
            assert np.all(np.abs(probs.sum(axis=1) - 1) < 1e-6)
            initial_loss = cross_entropy(Y_train, forward(X_train, initial)[1])
            assert abs(initial_loss - np.log(10)) < 0.01
            sanity.append(dict(width=width, initial_loss=initial_loss,
                               max_probability_sum_error=float(np.max(abs(probs.sum(axis=1)-1)))))
            for lr in [0.001, 0.01, 0.1]:
                p, history = train(X_train, Y_train, width, lr)
                histories.extend(history)
                probs, cache = forward(X_test, p)
                correct = int(np.sum(probs.argmax(axis=1) == y_test))
                summaries.append(dict(width=width, learning_rate=lr,
                    final_train_loss=history[-1]['train_loss'],
                    final_train_accuracy=history[-1]['train_accuracy'],
                    test_accuracy=correct/len(y_test), test_correct=correct,
                    test_count=len(y_test)))
                models[(width, lr)] = p
            # Check a trained snapshot so tiny initial derivatives do not make
            # relative-error comparisons dominated by floating-point cancellation.
            p = models[(width, 0.1)]
            rows = numerical_gradient_check(X_train[:32], Y_train[:32], p)
            for row in rows: row['width'] = width
            checks.extend(rows)
            assert sum(r['passed'] for r in rows) >= 9, rows
            timing = time_passes(X_train[:32], Y_train[:32], p)
            timing['width'] = width; timings.append(timing)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = ['#6689a8', '#dd963a', '#167d86']
    for ax, width in zip(axes, [16, 64]):
        for lr, color in zip([0.001, 0.01, 0.1], colors):
            h = [r for r in histories if r['width']==width and r['learning_rate']==lr]
            ax.plot([r['epoch'] for r in h], [r['train_loss'] for r in h],
                    label=f'learning rate = {lr}', color=color, linewidth=2)
        ax.set(title=f'Two hidden layers, W = {width}', xlabel='Epoch')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Mean training cross-entropy')
    fig.suptitle('Digits: 50 epochs, batch size 32, initial weight std 0.01')
    fig.tight_layout(); fig.savefig(out/'loss_curves.png', dpi=180); plt.close(fig)
    write_csv(out/'epoch_logs.csv', histories)
    write_csv(out/'gradient_check.csv', checks)
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'timing.csv', timings)
    result = dict(train_size=len(X_train), test_size=len(X_test),
        train_class_counts=np.bincount(y_train).tolist(),
        test_class_counts=np.bincount(y_test).tolist(), sanity=sanity,
        summary=summaries, gradient_checks=checks, timings=timings,
        configuration=dict(epochs=50, batch_size=32, split_seed=42,
            initialization_seed=42, shuffle_seed=123, gradient_seed=2026,
            epsilon=1e-5, dtype='float64', blas_threads=1),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
            platform=platform.platform(), processor=platform.processor()))
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nFINAL RESULTS\n'+json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    run_experiments()
