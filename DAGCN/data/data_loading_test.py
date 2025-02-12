import os
import numpy as np
import math
import torch
import torch_geometric
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)
import torch_geometric.transforms as transforms
from directed_heterophilous_graphs import DirectedHeterophilousGraphDataset
from mask import get_mask


def get_dataset(name: str, root_dir: str, undirected=False, self_loops=False, transpose=False):
    path = os.path.join(root_dir, "dataset")  # 数据集文件存放的目录
    evaluator = None

    if name in ["chameleon", "squirrel"]:
        # 加载 "chameleon" 或 "squirrel" 数据集
        dataset = WikipediaNetwork(root=path, name=name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)
    elif (name in ["directed_roman_empire", "directed_amazon_ratings", "directed_questions"]):
        # 加载 "directed-roman-empire" 数据集
        dataset = DirectedHeterophilousGraphDataset(name=name, transform=transforms.NormalizeFeatures(), root=path)
    elif name in ["cora_ml", "citeseer_full"]:
        if name == "citeseer_full":
            name = "citeseer"
        # 加载 CitationFull 数据集
        dataset = CitationFull(path, name)
    else:
        raise Exception("Unknown dataset.")

    # 处理图的预处理逻辑
    if undirected:
        dataset._data.edge_index = torch_geometric.utils.to_undirected(dataset._data.edge_index)
    if self_loops:
        dataset._data.edge_index, _ = torch_geometric.utils.add_self_loops(dataset._data.edge_index)
    if transpose:
        dataset._data.edge_index = torch.stack([dataset._data.edge_index[1], dataset._data.edge_index[0]])

    return dataset, evaluator


def get_dataset_split(name, data, split_number):
    if name in ["chameleon", "squirrel", "directed_roman_empire", "directed_amazon_ratings", "directed_questions"]:
        # 返回这些数据集的训练、验证和测试拆分
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )
    if name in ["cora_ml", "citeseer_full"]:
        # 对于 "cora_ml" 和 "citeseer_full"，使用 50/25/25 的拆分比例
        return set_uniform_train_val_test_split(split_number, data, train_ratio=0.5, val_ratio=0.25)
    else:
        raise Exception("Unknown dataset for splitting.")


def set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25):
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]

    # 一些节点的标签是 -1（即未标注），因此需要排除这些节点
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)

    idxs = list(range(num_labeled_nodes))
    # 对索引进行原地打乱
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train: num_train + num_val]
    test_idx = idxs[num_train + num_val:]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    train_mask = get_mask(train_idx, num_nodes)
    val_mask = get_mask(val_idx, num_nodes)
    test_mask = get_mask(test_idx, num_nodes)

    return train_mask, val_mask, test_mask


def main():
    # 设置根目录
    root_dir = "./"  # 替换为你本地的数据集根目录路径

    # 定义所有支持的数据集名称
    dataset_names = ["chameleon", "squirrel", "directed_roman_empire", "directed_amazon_ratings", "directed_questions", "cora_ml", "citeseer_full"]

    for dataset_name in dataset_names:
        print(f"正在处理数据集: {dataset_name}")

        # 获取数据集
        dataset, evaluator = get_dataset(dataset_name, root_dir)

        # 获取训练、验证、测试集的拆分
        split_number = 0  # 选择第一个拆分，假设数据集包含多个拆分
        train_mask, val_mask, test_mask = get_dataset_split(dataset_name, dataset._data, split_number)

        # 打印训练、验证、测试集的节点数量
        num_train_nodes = train_mask.sum().item()
        num_val_nodes = val_mask.sum().item()
        num_test_nodes = test_mask.sum().item()

        print(f"训练节点数量: {num_train_nodes}")
        print(f"验证节点数量: {num_val_nodes}")
        print(f"测试节点数量: {num_test_nodes}")

        # 输出训练、验证、测试节点的比例
        total_nodes = dataset._data.num_nodes
        print(f"总节点数量: {total_nodes}")
        print(f"训练节点比例: {num_train_nodes / total_nodes:.4f}")
        print(f"验证节点比例: {num_val_nodes / total_nodes:.4f}")
        print(f"测试节点比例: {num_test_nodes / total_nodes:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
