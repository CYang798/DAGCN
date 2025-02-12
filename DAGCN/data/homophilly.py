import os
import torch
from data_loading_test import get_dataset


def get_homophily(data):
    """计算边同质性和节点同质性."""
    edge_index = data.edge_index
    y = data.y.squeeze() if data.y.dim() > 1 else data.y

    # 边同质性
    edge_homophily = ((y[edge_index[0]] == y[edge_index[1]]).float().sum() / edge_index.size(1)).item()

    # 节点同质性
    num_nodes = data.num_nodes
    node_homophily = torch.zeros(num_nodes, dtype=torch.float)
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) > 0:
            node_homophily[i] = (y[neighbors] == y[i]).float().mean()
    node_homophily = node_homophily.mean().item()

    return edge_homophily, node_homophily


def analyze_homophily(name: str, root_dir: str, split_number=None, undirected=False, self_loops=False, transpose=False):
    """判断数据集是同质图还是异质图."""
    dataset, _ = get_dataset(name, root_dir, split_number, undirected, self_loops, transpose)
    data = dataset[0]

    edge_homophily, node_homophily = get_homophily(data)

    is_homophilic = edge_homophily > 0.5
    print(f"{name} 数据集:")
    print(f"  边同质性: {edge_homophily:.4f}")
    print(f"  节点同质性: {node_homophily:.4f}")
    print(f"  判定: {'同质图' if is_homophilic else '异质图'}")


# 示例用法
if __name__ == "__main__":
    root_dir = './dataset'

    datasets_to_check = [
        ('chameleon', 'WikipediaNetwork'),
        ('squirrel', 'WikipediaNetwork'),
        ('Cora', 'Planetoid'),
        ('Citeseer', 'Planetoid'),
        ('photo', 'Amazon'),
        ('film', 'Actor'),
        ('ogbn-arxiv', 'OGB'),
    ]

    for name, sub_dir in datasets_to_check:
        print(f"正在分析 {name} 数据集...")
        analyze_homophily(name, os.path.join(root_dir, sub_dir))
