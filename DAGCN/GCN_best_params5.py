import os
import torch
import numpy as np
import uuid
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
import pytorch_lightning as pl

from utils.utils import get_available_accelerator
from data.data_loading import get_dataset, get_dataset_split
from data.dataset import FullBatchGraphDataset
from model.GCN_improve5 import get_model, LightingFullBatchModelWrapper
from utils.arguments import args


def run(args):
    torch.manual_seed(0)

    # 检查使用的 GPU 数量
    if args.gpu_idx >= 0 and torch.cuda.is_available():
        num_gpus = min(torch.cuda.device_count(), 1)
        print(f"Using 1 GPU (GPU index: {args.gpu_idx}) for training.")
        devices = [args.gpu_idx]  # 使用 GPU 时传入 GPU 索引
    else:
        num_gpus = 0
        print("Using CPU for training.")
        devices = 1  # 使用 1 个 CPU 核心

    # 获取数据集和 DataLoader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    data_loader = DataLoader(FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0])

    val_accs, test_accs = [], []
    for num_run in range(args.num_runs):
        # 获取当前运行的训练/验证/测试集分割
        train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, num_run)

        # 获取模型
        args.num_features, args.num_classes = data.num_features, dataset.num_classes
        model = get_model(args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # 设置 PyTorch Lighting 回调
        early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        model_summary_callback = ModelSummary(max_depth=-1)
        if not os.path.exists(f"{args.checkpoint_directory}/"):
            os.mkdir(f"{args.checkpoint_directory}/")
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
        )

        # 设置 PyTorch Lightning 训练器
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[
                early_stopping_callback,
                model_summary_callback,
                model_checkpoint_callback,
            ],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=devices,  # 根据设备设置使用 GPU 或 CPU
        )

        # 训练模型
        trainer.fit(model=lit_model, train_dataloaders=data_loader)

        # 计算验证和测试准确率
        val_acc = model_checkpoint_callback.best_model_score.item()
        test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print(f"Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")

    return val_accs  # 返回验证集的准确率列表


def find():
    # 设置数据集名称
    dataset_choice = "chameleon"  # 在这里设置你想要使用的数据集名称
    args.dataset = dataset_choice

    # 根据数据集名称设置路径
    root_dir = './data/dataset'  # 数据集根目录

    # 直接使用已经下载和处理过的数据集路径
    if args.dataset in ['chameleon', 'squirrel', 'directed-roman-empire', 'cora_ml', 'citeseer_full']:
        args.dataset_directory = root_dir # 拼接数据集路径
    else:
        raise ValueError(f"未知的数据集: {args.dataset}")

    args.checkpoint_directory = f"./result/{dataset_choice}"

    # 预处理参数
    args.undirected = True  # 适配无方向性设置
    args.self_loops = False
    args.transpose = False

    # 训练参数
    args.num_epochs = 10000  # 减少轮数以加速
    args.num_runs = 10  # 减少运行次数以优化效率
    args.use_best_hyperparams = False

    # 系统参数
    args.gpu_idx = 0
    args.num_workers = 0
    args.log = "INFO"
    args.profiler = False

    # 定义超参数的搜索空间
    lr_vals = [0.001, 0.01]
    hidden_dim_vals = [64, 128, 256]
    num_layers_vals = [4, 5, 6]
    jk_vals = ["max", "cat"]
    patience_vals = [400]
    beta1_vals = [0.5, 0.7, 0.9]
    learn_alpha_vals = [False]
    normalize_vals = [True]
    conv_type_vals = ["dir-gcn"]

    # 初始化最佳结果
    best_val_acc = -float('inf')
    best_params = None

    # 遍历所有超参数组合
    for lr in lr_vals:
        for hidden_dim in hidden_dim_vals:
            for num_layers in num_layers_vals:
                for jk in jk_vals:
                    for patience in patience_vals:
                        for beta1 in beta1_vals:
                            for learn_alpha in learn_alpha_vals:
                                for normalize in normalize_vals:
                                    for conv_type in conv_type_vals:
                                        # 设置当前超参数
                                        args.lr = lr
                                        args.hidden_dim = hidden_dim
                                        args.num_layers = num_layers
                                        args.jk = jk
                                        args.patience = patience
                                        args.beta1 = beta1
                                        args.learn_alpha = learn_alpha
                                        args.normalize = normalize
                                        args.conv_type = conv_type

                                        # 运行模型并获取验证集准确率
                                        val_accs = run(args)
                                        max_val_acc = max(val_accs)  # 获取最大验证准确率

                                        # 更新最佳参数
                                        if max_val_acc > best_val_acc:
                                            best_val_acc = max_val_acc
                                            best_params = {
                                                "lr": lr,
                                                "hidden_dim": hidden_dim,
                                                "num_layers": num_layers,
                                                "jk": jk,
                                                "patience": patience,
                                                "beta1": beta1,
                                                "learn_alpha": learn_alpha,
                                                "normalize": normalize,
                                                "conv_type": conv_type,
                                            }

    # 输出最佳结果
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc}")


if __name__ == "__main__":
    find()
