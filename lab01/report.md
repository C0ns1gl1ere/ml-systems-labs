# Lab 1 Report

## 1. Goal
Establish an isolated Python environment with pinned dependencies, train baseline machine learning models on the Breast Cancer Wisconsin dataset, and measure their system footprints to evaluate deployment suitability across target hardware budgets.

## 2. Method
- Split the Breast Cancer Wisconsin dataset 70/30 stratified with random_state=42.
- Trained two baseline models: LogisticRegression (max_iter=1000, random_state=42) and RandomForestClassifier (n_estimators=100, random_state=42).
- Measured training time as the median of 5 runs following 1 warm-up run.
- Measured single-sample inference latency as the median of 100 runs.
- Recorded serialized model artifact size (KB) using joblib and tracked process peak memory usage.

## 3. Results
| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| Test Accuracy | 0.9415 | 0.9357 |
| Training Time (ms) | 291.11 | 94.37 |
| Inference Latency (ms) | 0.0317 | 1.3654 |
| Model Size (KB) | 1.03 | 284.07 |
| Peak Memory (MB) | 0.83 | 0.80 |

### Deployment Budget Fit:
- **LogisticRegression**: Meets the constraints for **TinyML**, **Mobile**, **Edge**, and **Cloud**, as its size (1.03 KB <= 100 KB) and latency (0.0317 ms <= 10 ms) comfortably fit within the strictest TinyML limits.
- **RandomForestClassifier**: Suitable for **Mobile**, **Edge**, and **Cloud**, but **incompatible with TinyML** because its artifact size (284.07 KB) exceeds the TinyML ceiling of 100 KB.

## 4. Conclusions
1. Logistic Regression provides minimal storage and latency overhead (1.03 KB, 0.03 ms), making it an optimal candidate for extreme edge and microcontroller environments (TinyML).
2. Random Forest achieves competitive accuracy (0.9357) and faster training, but storing 100 decision trees expands artifact size to 284 KB, restricting deployment to Mobile, Edge, or Cloud environments.
3. Warm-up executions combined with median metric aggregation successfully removed runtime jitter and produced stable measurements across training and inference benchmarks.
```[cite: 1, 3, 17]