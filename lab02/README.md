# Lab 2: NumPy neural network

- [report.md](report.md): report in Markdown, including formulas, results, the loss plot, and conclusions.
- `Lab2_report.pdf`: compact report with the architecture, gradient derivation, numerical checks, results, and timing analysis.
- [lab2.py](lab2.py): standalone NumPy implementation of the network, all six training runs, gradient checks, timing, and plotting.
- `results/`: required loss plot, 300 epoch logs, final metrics, 20 gradient checks, timings, and raw results.

## Reproduce
Use Python 3.12. From this folder, run:

```text
python -m pip install -r requirements.txt
python lab2.py
```

The script runs directly with Python and writes the experiment outputs to `results/`. Read `report.md` or the PDF to inspect the saved experiment without rerunning training.

`load_digits` is bundled with scikit-learn, so running the installed project does not download the dataset. Timings are machine-dependent. Split seed 42, weight seed 42, shuffle seed 123, numerical-check seed 2026. Training and timing use one BLAS thread.

The architecture and chain-rule derivation are included in the report.
