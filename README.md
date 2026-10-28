# ML systems labs

Standalone Python implementations of the laboratory experiments. Run the scripts from the working directories shown below; explanations and saved results are available in the Markdown and PDF reports.

| Lab | Python entry point | Report |
|---|---|---|
| 1: baseline classifiers | [lab01/src/main.py](lab01/src/main.py) | [lab01/report.md](lab01/report.md) |
| 2: neural network from scratch | [lab02/lab2.py](lab02/lab2.py) | [lab02/report.md](lab02/report.md) |
| 3: architecture costs | [lab03/lab3.py](lab03/lab3.py) | [lab03/report.md](lab03/report.md) |

## Lab 1

From the repository root, install the shared requirements, then run from `lab01/src` so outputs are saved to `lab01/results`:

```text
python -m pip install -r requirements.txt
cd lab01/src
python main.py
```

## Lab 2

Use Python 3.12. From the repository root:

```text
cd lab02
python -m pip install -r requirements.txt
python lab2.py
```

See [lab02/README.md](lab02/README.md) for experiment details.

## Lab 3

Use Python 3.11. From the repository root:

```text
cd lab03
python -m pip install -r requirements.txt
python lab3.py --data data --threads 4
```

The first run downloads Fashion-MNIST. See [lab03/README.md](lab03/README.md) for experiment details.

Lab 2 and Lab 3 regenerate their experiment outputs in their respective `results/` folders. Lab 3 also updates its Markdown report. The saved PDF reports are snapshots of the submitted experiments.
