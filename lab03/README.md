# Lab 3: PyTorch architectures and costs

- [report.md](report.md): full report with hand calculations, profiling comparison, latency protocol, and conclusions.
- [Lab3_report.pdf](Lab3_report.pdf): compact report with tables, formulas, and both trade-off plots.
- [src/lab3.py](src/lab3.py): standalone PyTorch implementation of all three specified architectures.
- [evidence/hand_calculations.jpg](evidence/hand_calculations.jpg): original handwritten MLP and CNN calculations, extracted from the supplied scan.
- `results/`: metrics, 15 epoch logs, plots, torchinfo summaries, environment versions, and final model weights.

## Reproduce

Use Python 3.11. From this folder:

```text
python -m pip install -r requirements.txt
python src/lab3.py --data data --threads 4
```

The first run downloads Fashion-MNIST. Each model trains for five epochs with Adam, learning rate 0.001, and batch size 128. Seed 42 is used for model initialization, the 55k/5k training split, and data shuffling. The test set is evaluated after training. Running the program updates `results/` and `report.md`; the PDF is a snapshot of the submitted experiment.

The script runs directly with Python. Read `report.md` or the PDF to inspect the saved experiment without rerunning training. Exact package versions from this experiment are in `results/environment.txt`; timings and numerical results can vary by computer and software version.

`src/` contains the Python source; `evidence/` contains the handwritten scan. Generated outputs and the Markdown report are saved in `lab03/results/` and `lab03/report.md` even when the script is launched from another working directory. Without `--data`, the dataset cache defaults to `lab03/data/`.

The explicit architectures conflict with the manual's 100k-150k parameter target. This submission preserves the specified layers and explains the discrepancy in the report. The best measured test accuracy is CNN: 89.64%; MLP is the fastest in this CPU run.
