import os
import time
import joblib
import psutil
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

os.makedirs("../results", exist_ok=True)
csv_path = "../results/baseline_accuracy.csv"

with open(csv_path, "w") as f:
    f.write("Model,Accuracy,Train_Time_ms,Inference_Latency_ms,Size_KB,Peak_Mem_MB\n")
    for name, model in models.items():
        # Замер пиковой памяти и времени обучения (1 warm-up + 5 прогонов)
        mem_before = psutil.Process().memory_info().rss
        model.fit(X_train, y_train)  # warm-up
        train_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_times.append((time.perf_counter() - t0) * 1000)
        median_train_time = np.median(train_times)
        mem_after = psutil.Process().memory_info().rss
        peak_mem = max(0.0, (mem_after - mem_before) / (1024 * 1024))

        # Тестовая точность (4 знака после запятой)
        acc = accuracy_score(y_test, model.predict(X_test))

        # Замер латентности инференса одного сэмпла (100 прогонов)
        sample = X_test[0:1]
        model.predict(sample)  # warm-up
        inf_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(sample)
            inf_times.append((time.perf_counter() - t0) * 1000)
        median_inf_time = np.median(inf_times)

        # Сохранение и размер модели
        model_path = f"../results/{name}_model.joblib"
        joblib.dump(model, model_path)
        size_kb = os.path.getsize(model_path) / 1024

        f.write(f"{name},{acc:.4f},{median_train_time:.2f},{median_inf_time:.4f},{size_kb:.2f},{peak_mem:.2f}\n")

print("Готово! Данные сохранены в results/baseline_accuracy.csv")
