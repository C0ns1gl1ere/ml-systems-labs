# Individual project proposal

**Project** · CPU deployment of a quantized clothing image classifier

## Problem and dataset

Build a small offline classifier that predicts one of ten clothing categories from a 28 × 28 grayscale image. The intended use is a local sorting demo for standardized garment thumbnails. I will use Fashion-MNIST, with the same seeded split as Lab 3 (55,000 training and 5,000 validation examples) and the official 10,000-example held-out test set. The 20,490-parameter Lab 3 CNN is the FP32 baseline. Lab 3 recorded 89.64% test accuracy.

## Success criteria

The deployed model must reach at least 89% test top-1 accuracy and stay within one percentage point of the matched FP32 baseline. I will also report macro F1 and a confusion matrix to identify categories that lose accuracy after optimization. The exported model artifact must occupy at most 5,000,000 bytes. End-to-end p95 latency, including normalization and classification of one image, must be at most 50 ms on a named x86-64 CPU with one inference thread, without a GPU or network dependency. I will report cold-start latency separately.

## Planned optimization and evaluation

Train and select the FP32 CNN using validation data. Apply static INT8 post-training quantization to the convolutional and linear layers using a representative, class-balanced subset of 1,000 training images for calibration. Confirm that the exported graph uses quantized operators and measure accuracy and latency against FP32 in the same runtime. Use 100 warm-up inferences and 1,000 timed batch-of-one requests with identical preprocessing. If post-training quantization misses the accuracy target, use quantization-aware fine-tuning on the training split and select the checkpoint on validation data before the final test evaluation.

## Deliverables

A reproducible training and export command, the serialized model, and a results table with artifact size, median and p95 latency, top-1 accuracy, and macro F1. The demo assumes a centered grayscale input. Performance on arbitrary photos is outside the scope of this project.
