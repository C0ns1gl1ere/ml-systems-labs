# ML systems labs

Standalone Python implementations of the laboratory experiments. Run the scripts from the working directories shown below; explanations and saved results are available in the Markdown and PDF reports.

| Lab | Python entry point | Report |
|---|---|---|
| 1: baseline classifiers | [lab01/src/main.py](lab01/src/main.py) | [lab01/report.md](lab01/report.md) |
| 2: neural network from scratch | [lab02/lab2.py](lab02/lab2.py) | [lab02/report.md](lab02/report.md) |
| 3: architecture costs | [lab03/src/lab3.py](lab03/src/lab3.py) | [lab03/report.md](lab03/report.md) |
| 4: data pipeline and quality | [lab04/run.py](lab04/run.py) | [lab04/REPORT.md](lab04/REPORT.md) |
| 5: frameworks and automatic differentiation | [lab05/lab05.py](lab05/lab05.py) | [lab05/REPORT_UA.md](lab05/REPORT_UA.md) |

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
python src/lab3.py --data data --threads 4
```

The first run downloads Fashion-MNIST. See [lab03/README.md](lab03/README.md) for experiment details.

## Lab 4

Use Python 3.11. From the repository root:

```text
cd lab04
python -m pip install -r requirements-lock.txt
python run.py
python verify.py
```

On Windows, `cd lab04` followed by `.\run_lab.ps1` sets up an environment and runs the experiment. Raw Adult data, processed data, fitted models, validation reports and reproducibility evidence are included. See [lab04/README.md](lab04/README.md) and [lab04/Lab4_Adult.ipynb](lab04/Lab4_Adult.ipynb). The 50-row human dictionary review remains pending in `lab04/results/audit_50_manual.csv`.

Lab 2 and Lab 3 regenerate their experiment outputs in their respective `results/` folders. Lab 3 also updates its Markdown report. The saved PDF reports are snapshots of the submitted experiments.

## Lab 5

Use Python 3.11 or newer. From the repository root:

```text
cd lab05
python -m pip install -r requirements.txt
python lab05.py
```

Includes a scalar automatic-differentiation engine, gradient checks against PyTorch and finite differences, five-run training benchmarks on make_moons, eager versus traced inference, and tensor contiguity measurements. See [lab05/README.md](lab05/README.md), [the notebook](lab05/lab05.ipynb), and [the Ukrainian report](lab05/REPORT_UA.md). Saved measurements and the training plot are included; environment.txt records the tested dependency versions.
