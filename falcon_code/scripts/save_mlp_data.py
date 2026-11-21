import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import torch
import joblib

from data_modules.circuit_dataset import CircuitGraphDataset
from utils.data_utils import PerformanceToClassWithPerMetricScaler
from utils.io_tools import load_yaml


if __name__ == "__main__":
    root_dir = "dataset"
    config = load_yaml(os.path.join("config", "data_config.yaml"))
    global_perf_list = list(config["Performance"].keys())   # 只拿key
    circuit_to_code = config["Classes"]  # 全拿，是个字典

    print("[INFO] Loading base dataset...")
    full_dataset = CircuitGraphDataset(root_dir, circuit_to_code, global_perf_list)
    print(f"[INFO] Total samples in dataset: {len(full_dataset)}")

    print("[INFO] Loading splits from data_splits.npz...")
    splits = np.load(os.path.join(root_dir, "data_splits.npz"))
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    print("[INFO] Loaded split indices from data_splits.npz")

    print("[INFO] Creating + fitting scaler on train set...")
    train_dataset = PerformanceToClassWithPerMetricScaler(full_dataset, train_idx)
    # 这里的scalar包含对于train_dataset的mean, var, std/scale, number of items
    scaler = train_dataset.scaler
    # 就是单纯保存文件到performance_scaler_mlp.pkl
    joblib.dump(scaler, os.path.join(root_dir, "performance_scaler_mlp.pkl"))
    print("[INFO] Saved scaler to performance_scaler_mlp.pkl")

    print("[INFO] Creating validation and test sets...")
    val_dataset = PerformanceToClassWithPerMetricScaler(full_dataset, val_idx, scaler=scaler)
    test_dataset = PerformanceToClassWithPerMetricScaler(full_dataset, test_idx, scaler=scaler)

    # Convert to tensors
    # 对应单个样本，这里的x,y分别是标准化后的16个perf metrics 和对应的circuit_type_code
    train_samples = [(x, y) for x, y in train_dataset]
    val_samples = [(x, y) for x, y in val_dataset]
    test_samples = [(x, y) for x, y in test_dataset]

    torch.save({
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }, os.path.join(root_dir, "performance_class_data.pt"))

    print("[INFO] Saved preprocessed MLP dataset to performance_class_data.pt")