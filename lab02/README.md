# Lab 2 submission package

- `Lab2.ipynb`: self-contained notebook with executed outputs, formulas, all six runs, checks, and plot.
- `Lab2_report.pdf`: compact report with the architecture, gradient derivation, numerical checks, results, and timing analysis.
- `lab2.py`: equivalent standalone NumPy implementation.
- `results/`: required loss plot, 300 epoch logs, final metrics, 20 gradient checks, timings, raw results, and two trained snapshots.

## Reproduce
Use Python 3.12. From this folder, run:

```text
python -m pip install -r requirements.txt
python lab2.py
```

Or open `Lab2.ipynb` in Jupyter/VS Code and run all cells with those dependencies installed. The notebook already contains outputs. Jupyter itself is not needed to run `lab2.py`.

`load_digits` is bundled with scikit-learn, so running the installed project does not download the dataset. Timings are machine-dependent. Split seed 42, weight seed 42, shuffle seed 123, numerical-check seed 2026. Training and timing use one BLAS thread.

The hand-written parts requested in the manual are typeset in the notebook and report. If literal handwriting is required, transcribe the architecture and chain-rule derivation. Add your student details if your course requires them.
