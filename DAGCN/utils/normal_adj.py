from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm



# `get_norm_adj`函数根据选择的规范化方法（"sym", "row", "dir"）对邻接矩阵进行归一化。
def get_norm_adj(adj, norm):
    if norm == "sym":  # 对称归一化
        return gcn_norm(adj, add_self_loops=False)  # 使用GCN的规范化方法
    elif norm == "row":  # 行归一化
        return row_norm(adj)  # 调用行归一化函数
    elif norm == "dir":  # 有向图归一化
        return directed_norm(adj)  # 调用有向图归一化函数
    else:
        raise ValueError(f"{norm} normalization is not supported")  # 如果传入了无效的归一化类型，抛出异常


# `row_norm`函数对邻接矩阵进行行归一化处理
def row_norm(adj):
    """
    应用行归一化:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)  # 计算每一行的和，即每个节点的出度 \mathbf{D}_{out}
    return mul(adj, 1 / row_sum.view(-1, 1))  # 用出度的倒数对邻接矩阵进行逐元素相乘归一化


# `directed_norm`函数对有向图邻接矩阵进行归一化处理
def directed_norm(adj):
    """
    应用有向图的归一化:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}
    """
    in_deg = sparsesum(adj, dim=0)  # 计算入度，即列和 \mathbf{D}_{in}
    in_deg_inv_sqrt = in_deg.pow_(-0.5)  # 计算入度的平方根倒数 \mathbf{D}_{in}^{-1/2}
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)  # 避免出现无穷大的情况

    out_deg = sparsesum(adj, dim=1)  # 计算出度，即行和 \mathbf{D}_{out}
    out_deg_inv_sqrt = out_deg.pow_(-0.5)  # 计算出度的平方根倒数 \mathbf{D}_{out}^{-1/2}
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)  # 避免出现无穷大的情况

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))  # 根据出度的平方根倒数对邻接矩阵进行归一化
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))  # 根据入度的平方根倒数对邻接矩阵进行归一化
    return adj  # 返回归一化后的邻接矩阵
