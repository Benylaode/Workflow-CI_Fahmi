#!/usr/bin/env python3
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn


mlflow.sklearn.autolog()


# Atur tracking URI
if os.getenv("GITHUB_ACTIONS") == "true":
    MLFLOW_TRACKING_URI = "file:./mlruns"
    print("🔧 Running in GitHub Actions → using local MLflow store:", MLFLOW_TRACKING_URI)
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    print("🏠 Running locally → using MLflow Server:", MLFLOW_TRACKING_URI)


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "california_housing_exp")

DATA_DIR = "california_housing_data/namadataset_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

OUTPUT_DIR = "california_housing_data/artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("📂 Memuat dataset...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET_COL = "MedHouseVal"

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]


# Mulai MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"🔍 MLflow tracking: {MLFLOW_TRACKING_URI}")


# Daftar model
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}


def train_model(name, model):
    print(f"\n🚀 Training model: {name}")
    t0 = time.time()

    model.fit(X_train, y_train)
    duration = time.time() - t0

    pred = model.predict(X_test)

    # Hitung metrik
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    # Mulai MLflow Run
    with mlflow.start_run(run_name=name):
        # Autolog otomatis mencatat parameter, metrik, model, artifact, dll
        print(f"📌 Autolog aktif → logging otomatis untuk model {name}")

        # Simpan plot prediksi sebagai artifact tambahan
        plt.figure()
        plt.scatter(y_test, pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} Prediction Plot")

        plot_path = os.path.join(OUTPUT_DIR, f"{name}_plot.png")
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

    return {"model": name, "mse": mse, "mae": mae, "r2": r2}


def main():
    results = []
    for name, model in models.items():
        results.append(train_model(name, model))

    # Simpan benchmark
    results_df = pd.DataFrame(results)
    benchmark_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(benchmark_path, index=False)

    print("\n📊 Hasil training disimpan → benchmark_results.csv")

    with mlflow.start_run(run_name="benchmark_summary"):
        mlflow.log_artifact(benchmark_path)


if __name__ == "__main__":
    main()
