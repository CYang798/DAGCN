import torch
from torch import nn, optim
from torch_sparse import SparseTensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge

from utils.normal_adj import get_norm_adj


def get_conv(conv_type, input_dim, output_dim, alpha1, alpha2, beta1):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha1, alpha2, beta1)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


class DirGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha1, alpha2, beta1):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha1 = alpha1  # 一阶特征权重
        self.alpha2 = alpha2  # 二阶特征权重
        self.beta1 = beta1  # 一阶传播的整体权重，二阶由 (1 - beta1) 决定
        self.adj_norm, self.adj_t_norm = None, None
        self.adj_second_order_in_norm, self.adj_second_order_out_norm = None, None

    def compute_second_order_proximity(self, edge_index, num_nodes):
        """计算2跳邻接矩阵 A^2 和 A转置的平方 (A^T * A^T)"""
        row, col = edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))

        # 使用 matmul 计算二阶接近性 A^2
        adj_square = adj.matmul(adj)  # A * A
        adj_square = adj_square.set_diag(0)  # 移除对角线元素
        self.adj_second_order_in_norm = get_norm_adj(adj_square, norm="dir")

        # 使用 matmul 计算 A转置的平方 A^T * A^T
        adj_t_square = adj_t.matmul(adj_t)  # A^T * A^T
        adj_t_square = adj_t_square.set_diag(0)  # 移除对角线元素
        self.adj_second_order_out_norm = get_norm_adj(adj_t_square, norm="dir")

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

            # 计算并归一化二阶邻接矩阵 A^2 和 A^T * A^T
            self.compute_second_order_proximity(edge_index, num_nodes)

        # 一阶传播
        first_order_src_to_dst = self.lin_src_to_dst(self.adj_norm @ x)
        first_order_dst_to_src = self.lin_dst_to_src(self.adj_t_norm @ x)

        # 二阶传播
        second_order_src = self.lin_src_to_dst(self.adj_second_order_out_norm @ x)
        second_order_dst = self.lin_dst_to_src(self.adj_second_order_in_norm @ x)

        # 加权
        return self.beta1 * (self.alpha1 * first_order_src_to_dst + (1 - self.alpha1) * first_order_dst_to_src) + \
               (1 - self.beta1) * (self.alpha2 * second_order_src + (1 - self.alpha2) * second_order_dst)


class GNN(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_dim,
                 num_layers=2,
                 dropout=0,
                 conv_type="dir-gcn",
                 jumping_knowledge=False,
                 normalize=False,
                 alpha1=1 / 2,
                 alpha2=1 / 2,
                 beta1=0.7,
                 learn_alpha=False):
        super(GNN, self).__init__()

        self.alpha1 = nn.Parameter(torch.ones(1) * alpha1, requires_grad=learn_alpha)
        self.alpha2 = nn.Parameter(torch.ones(1) * alpha2, requires_grad=learn_alpha)
        self.beta1 = nn.Parameter(torch.ones(1) * beta1, requires_grad=learn_alpha)

        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha1, self.alpha2, self.beta1)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha1, self.alpha2, self.beta1)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha1, self.alpha2, self.beta1))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha1, self.alpha2, self.beta1))

        if jumping_knowledge is not None:
            if jumping_knowledge == "cat":
                input_dim = hidden_dim * num_layers
            else:
                input_dim = hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)

class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        val_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", val_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer



def get_model(args):
    return GNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        beta1=args.beta1,
        learn_alpha=args.learn_alpha
    )
