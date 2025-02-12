import numpy as np
import math
import torch
import torch_geometric
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)
import torch_geometric.transforms as transforms

import sys
sys.path.append('data')
from directed_heterophilous_graphs import DirectedHeterophilousGraphDataset
from mask import get_mask


def get_dataset(name: str, root_dir: str, undirected=False, self_loops=False, transpose=False):
    path = root_dir  # 数据集文件存放的目录
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