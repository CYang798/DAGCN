import os
import torch
from torch_geometric.utils import to_undirected


def compute_unidirectional_edges_ratio(edge_index):
    """
    计算单向边占无向边总数的比例。
    :param edge_index: Tensor, 边的索引 (2, num_edges)
    :return: 单向边的比例 (百分比)
    """
    num_directed_edges = edge_index.shape[1]  # 原始图的边数
    num_undirected_edges = to_undirected(edge_index).shape[1]  # 转化为无向图后的边数
    num_unidirectional = num_undirected_edges - num_directed_edges  # 单向边的数量
    return (num_unidirectional / (num_undirected_edges / 2)) * 100  # 返回单向边的百分比


def process_pt_file(pt_file_path):
    """
    处理指定的 .pt 文件，计算并输出单向边的比例。
    :param pt_file_path: .pt 文件路径
    """
    # 加载 .pt 文件
    graph_data = torch.load(pt_file_path)

    # 如果是元组，提取第一个元素（Data对象）
    if isinstance(graph_data, tuple):
        graph_data = graph_data[0]  # 假设图数据是元组中的第一个元素

    # 获取边的索引并计算单向边比例
    edge_index = graph_data['edge_index']
    unidirectional_ratio = compute_unidirectional_edges_ratio(edge_index)
    print(f"{pt_file_path} - 单向边比例: {unidirectional_ratio:.2f}%")


def main():
    root_dir = "D:/桌面/DAGNN/data/dataset"  # 修改为你的根目录路径
    datasets = [
        "cora_ml", "citeseer_full", "chameleon", "squirrel", "directed_roman_empire", "directed_amazon_ratings", "directed_questions"
    ]

    for dataset_name in datasets:
        # 针对 chameleon 和 squirrel 需要调整路径
        if dataset_name in ["chameleon", "squirrel"]:
            # 在 chameleon 和 squirrel 数据集中，.pt 文件路径有一个中间文件夹 "geom_gcn"
            processed_path = os.path.join(root_dir, dataset_name, "geom_gcn", "processed")
        else:
            # 其他数据集的路径
            processed_path = os.path.join(root_dir, dataset_name, "processed")

        # 尝试加载不同的 .pt 文件
        pt_file_paths = [
            os.path.join(processed_path, "data.pt"),
            os.path.join(processed_path, "data_undirected.pt")
        ]

        pt_file_path = None
        for path in pt_file_paths:
            if os.path.exists(path):
                pt_file_path = path
                break

        if pt_file_path:
            process_pt_file(pt_file_path)
        else:
            print(f"{dataset_name} 数据集的 .pt 文件不存在！")


if __name__ == "__main__":
    main()
