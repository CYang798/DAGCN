import torch

# 定义生成掩码的函数
def get_mask(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = True
    return mask