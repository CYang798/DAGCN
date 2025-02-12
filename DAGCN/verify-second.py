import torch
from torch_sparse import SparseTensor

# 定义邻接边，构建图，节点编号从1到8
edge_index = torch.tensor([
    [1, 1, 1, 2, 2, 5, 5, 6],  # 起始节点
    [3, 4, 8, 3, 4, 2, 6, 7]   # 目标节点
])

num_nodes = 8  # 节点数量（节点编号从 1 到 8）

# 创建稀疏邻接矩阵 A 和转置矩阵 A^T
row, col = edge_index - 1  # 将节点编号减去1以适应0到7的范围
adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))

# 输出邻接矩阵 A 和转置矩阵 A^T
print("邻接矩阵 A:")
print(adj.to_dense())

print("\n转置矩阵 A^T:")
print(adj_t.to_dense())

# 计算二阶入度相似性（共享的入度邻居）
second_order_in = adj_t.matmul(adj)  # A^T * A
mask_t = adj_t.to_dense().to(torch.bool)  # 使用 A^T 生成一阶邻居的掩码
second_order_in = second_order_in.to_dense()  # 转为 dense 以便应用掩码
second_order_in.masked_fill_(mask_t, 0)  # 将一阶邻居设置为 0
second_order_in.fill_diagonal_(0)  # 去除自环

# 计算二阶出度相似性（共享的出度邻居）
second_order_out = adj.matmul(adj_t)  # A * A^T
mask = adj.to_dense().to(torch.bool)  # 使用 A 生成一阶邻居的掩码
second_order_out = second_order_out.to_dense()  # 转为 dense 以便应用掩码
second_order_out.masked_fill_(mask, 0)  # 将一阶邻居设置为 0
second_order_out.fill_diagonal_(0)  # 去除自环

# 输出结果
print("\n二阶入度相似性矩阵 (A^T * A):")
print(second_order_in)

print("\n二阶出度相似性矩阵 (A * A^T):")
print(second_order_out)
