import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from sklearn.model_selection import train_test_split
import numpy as np

from data_modules.circuit_dataset import CircuitGraphDataset
from utils.io_tools import load_yaml, seed_everything


if __name__ == "__main__":
    seed_everything(42)
    print("[INFO] Splitting dataset indices into train, validation, and test...")

    root_dir = "dataset"
    data_config = load_yaml('./config/data_config.yaml')
    global_perf_list = list(data_config['Performance'].keys())
    circuit_to_code = data_config['Classes']

    # Load full dataset
    # 初始化CircuitGraphDataset这个的时候会直接读取全部类型下所有拓扑的data数据，当然必须同时存在 graph.json 和 dataset.csv
    # (
    #     graph_file,  # 对应拓扑的 graph.json 路径
    #     df.iloc[idx],  # CSV 中的某一行完整数据
    #     subfolder  # 拓扑名称，如 CGLNA
    # )
    # full_dataset[i]会自动调用get
    full_dataset = CircuitGraphDataset(root_dir, circuit_to_code, global_perf_list)
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset[i].circuit_type_code for i in all_indices]  # 返回各个circuit_type_code的数组

    # First split: train (70%) vs temp (30%)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.3,
        stratify=all_labels,    # 会告诉 train_test_split()：划分时尽量维持每个电路拓扑原来的样本比例。
        random_state=42
    )

    # Second split: validation (15%) vs test (15%) from temp
    temp_labels = [full_dataset[i].circuit_type_code for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    # After splitting
    # np.savez()是NumPy提供的一个函数，用来把多个数组保存到同一个文件中。
    # 而.npz就是NumPy用来存储多个数组的一种文件格式。
    np.savez(f"./{root_dir}/data_splits.npz", train=train_idx, val=val_idx, test=test_idx)
    print("[INFO] Saved split indices to dataset/data_splits.npz")