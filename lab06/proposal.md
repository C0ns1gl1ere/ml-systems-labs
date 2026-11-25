# Image-Based Classification of Military Vehicles Using a Lightweight Neural Network

**Student name** [Fill in on the submission copy]  
**Student ID** [Fill in on the submission copy]

## Problem and intended use

Build an offline image classifier for a museum kiosk or classroom demo. A visitor selects a curated image showing one vehicle or scale model, and the kiosk displays its category immediately: tank, military truck, fixed-wing military aircraft, or military helicopter. A low-confidence prediction prompts the visitor to choose a clearer image.

## Dataset and training pipeline

Use [Military and Civilian Vehicles Classification, version 1](https://data.mendeley.com/datasets/njdjkbxdpn/1) by Gupta et al. (CC BY 4.0, 6,772 images across the full dataset). Retain the four military classes. Use the existing annotations to obtain single-vehicle images, and review the labels for ambiguous examples and illustrations. Group duplicates, near-duplicates, and crops from the same original photo before making a stratified train, validation, and test split so related images stay in the same partition.

Resize and normalize inputs consistently. Apply light augmentation only to training data. Train a small convolutional neural network with cross-entropy and Adam, log settings and metrics, and keep the checkpoint with the best validation macro F1. Fix the split and training seeds for reproducibility. Apply INT8 post-training quantization using training images for calibration. Choose the confidence threshold on validation data, then evaluate the FP32 and INT8 models on the same held-out test set.

## Success criteria and deployment

Target at least **0.80 macro F1** on the held-out test set and a drop of at most **0.02 macro F1** from FP32 to INT8. Report per-class recall and a confusion matrix to show which categories are hardest to distinguish. The exported model must be **at most 5 MB**. A single-image request on an x86-64 CPU with one inference thread must take **at most 50 ms at p95**, including resizing, normalization, prediction, and label selection. Time the warmed-up pipeline with images already in memory and report model loading separately. The kiosk runs locally and shows one result per visitor selection.
