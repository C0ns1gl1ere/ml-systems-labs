# Lab 3 submission package

- [Lab3.ipynb](Lab3.ipynb): notebook with saved results, formulas, plots, and an optional training command. The recorded outputs read the completed experiment; they do not claim the notebook itself trained the models.
- [report.md](report.md): full report with hand calculations, profiling comparison, latency protocol, and conclusions.
- [Lab3_report.pdf](Lab3_report.pdf): compact report with tables, formulas, and both trade-off plots.
- `lab3.py`: standalone PyTorch implementation of all three specified architectures.
- `results/`: metrics, 15 epoch logs, plots, torchinfo summaries, environment versions, and final model weights.

## Reproduce

Use Python 3.11. From this folder:

```text
python -m pip install -r requirements.txt
python lab3.py --data data --threads 4
```

The first run downloads Fashion-MNIST. Each model trains for five epochs with Adam, learning rate 0.001, and batch size 128. Seed 42 is used for model initialization, the 55k/5k training split, and data shuffling. The test set is evaluated after training. Running the program updates `results/` and `report.md`; the PDF is a snapshot of the submitted experiment.

For the notebook, install Jupyter separately if needed and run its cells from `lab03/`. Leave `RUN_TRAINING = False` to inspect saved results; set it to `True` to repeat training. Exact package versions from this experiment are in `results/environment.txt`; timings and numerical results can vary by computer and software version.

The explicit architectures conflict with the manual's 100k-150k parameter target. This submission preserves the specified layers and explains the discrepancy in the report. The best measured test accuracy is CNN: 89.64%; MLP is the fastest in this CPU run.
