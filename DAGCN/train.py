import os
import numpy as np
import uuid

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from utils.utils import use_best_hyperparams, get_available_accelerator
from data.data_loading import get_dataset, get_dataset_split
from data.dataset import FullBatchGraphDataset
from model.model import get_model, LightingFullBatchModelWrapper
from utils.arguments import args


def run(args, dataset_name):
    torch.manual_seed(0)

    # 获取数据集和dataloader
    dataset, evaluator = get_dataset(
        name=dataset_name,
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
        train_mask, val_mask, test_mask = get_dataset_split(dataset_name, data, num_run)

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

        # 设置Pytorch Lighting回调
        early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        model_summary_callback = ModelSummary(max_depth=-1)
        if not os.path.exists(args.checkpoint_directory):
            os.makedirs(args.checkpoint_directory)
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
        )

        # 设置Pytorch Lighting训练器
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[early_stopping_callback, model_summary_callback, model_checkpoint_callback],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=[args.gpu_idx],
        )

        # 训练模型
        trainer.fit(model=lit_model, train_dataloaders=data_loader)

        # 计算验证和测试准确率
        val_acc = model_checkpoint_callback.best_model_score.item()
        test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print(f"Dataset: {dataset_name} | Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")
    return np.mean(test_accs), np.std(test_accs)


if __name__ == "__main__":
    # 数据集根目录
    root_dir = './data/dataset'  # 数据集根目录

    # 设置要运行的数据集列表
    dataset_list = [
        'chameleon',
        # 'squirrel',
        # 'directed-roman-empire',
        #  'cora_ml',
        # 'citeseer_full'
    ]

    # 结果保存路径
    result_dir = './result'

    # 预处理参数
    args.undirected = True
    args.self_loops = False
    args.transpose = False

    # 训练参数
    args.num_epochs = 10000
    args.num_runs = 10
    args.weight_decay = 0.0

    # 系统参数设置
    args.use_best_hyperparams = True
    args.gpu_idx = 0
    args.num_workers = 0
    args.log = "INFO"
    args.profiler = False

    # 模型参数
    args.model = "gnn"
    args.hidden_dim = 64
    args.num_layers = 3
    args.dropout = 0.0
    args.alpha = 0.5
    args.learn_alpha = False
    args.conv_type = "sage"
    args.normalize = False
    args.jk = None

    # 循环处理每个数据集
    for dataset_choice in dataset_list:
        print(f"Running on dataset: {dataset_choice}")

        # 设置数据集路径
        args.dataset = dataset_choice
        args.dataset_directory = os.path.join(root_dir, dataset_choice)  # 拼接数据集路径
        args.checkpoint_directory = os.path.join(result_dir, dataset_choice)  # 设置结果保存路径

        # 加载最佳参数（如果use_best_hyperparams为True）
        args = use_best_hyperparams(args, dataset_choice) if args.use_best_hyperparams else args

        # 运行当前数据集的训练与测试
        mean_test_acc, std_test_acc = run(args, dataset_choice)

        # 保存测试结果到文件
        with open(os.path.join(result_dir, 'results.txt'), 'a') as f:
            f.write(f"{dataset_choice} | Test Acc: {mean_test_acc} +- {std_test_acc}\n")

    print("All datasets have been processed.")
