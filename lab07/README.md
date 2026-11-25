# Lab 07 · Compression through pruning and distillation

- [Report and efficiency comparison](report.md)
- [Accuracy–sparsity curves](results/sparsity.svg)
- [Complete measurements](results/metrics.csv) and [run configuration](results/environment.json)

Run from this directory with Python 3.11:

```sh
python -m pip install -r requirements.txt
python study.py --data data --out results-reproduced
```

The script downloads Fashion-MNIST on the first run. If you already have it, point `--data` at the directory containing `FashionMNIST/raw`. Use `--threads` to set the CPU thread count. Training defaults to CPU. Set `--device cuda` for CUDA training if your PyTorch installation supports it. Inference latency is always measured on CPU.

The default run trains on all 60,000 training examples and evaluates on the 10,000 official test examples. It uses five epochs for the baseline and each student and three fine-tuning epochs per pruning run. The output directory receives the complete CSV, environment metadata, and sparsity curves. Dense and CSR serialization sizes are measured using temporary files.
