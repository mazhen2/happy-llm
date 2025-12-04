"""
监督微调(SFT)训练脚本（使用DataParallel）

本脚本用于语言模型的监督微调阶段，支持多GPU并行训练。

主要功能：
1. 加载预训练模型权重
2. 使用SFTDataset加载对话格式的训练数据
3. 使用混合精度训练（bfloat16/float16）加速训练
4. 支持梯度累积，模拟更大的batch size
5. 使用余弦退火学习率调度策略
6. 支持多GPU并行训练（DataParallel）
7. 集成SwanLab进行实验跟踪和可视化

训练流程：
1. 加载预训练模型权重
2. 初始化模型和分词器
3. 创建数据加载器
4. 设置优化器和学习率调度
5. 执行训练循环（前向传播、反向传播、参数更新）
6. 定期保存模型检查点

注意：SFT训练只对assistant回复部分计算损失，忽略system和user消息。
"""

import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import SFTDataset

import swanlab

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


def Logger(content):
    """
    简单的日志记录函数

    Args:
        content (str): 要打印的日志内容
    """
    print(content)

def get_lr(it, all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略

    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率

    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数

    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率，为初始学习率的1/10

    # 1) Warmup阶段：线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters

    # 2) 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr

    # 3) 余弦退火阶段：使用余弦函数衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦系数
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    训练一个epoch的函数

    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算（只对assistant回复计算损失）
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存

    Args:
        epoch (int): 当前epoch编号
    """
    start_time = time.time()  # 记录开始时间

    # 遍历数据加载器中的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备（GPU/CPU）
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于只对assistant回复计算损失

        # ==================== 学习率调度 ====================
        # 计算当前步骤的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 前向传播 ====================
        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            out = model(X, Y)
            # 计算损失并除以累积步数（用于梯度累积）
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask展平为一维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（只对assistant回复部分计算损失）
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # ==================== 反向传播 ====================
        # 使用scaler进行混合精度的反向传播
        scaler.scale(loss).backward()

        # ==================== 参数更新 ====================
        # 每accumulation_steps步执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步骤
            scaler.step(optimizer)
            # 更新scaler的缩放因子
            scaler.update()

            # 清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)

        # ==================== 日志记录 ====================
        # 每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 如果启用SwanLab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # ==================== 模型保存 ====================
        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点文件名
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # 处理多卡保存：如果是DataParallel模型，需要访问.module属性
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式

        # 每20000步保存一个带步数标记的检查点
        if (step + 1) % 20000 == 0:
            model.eval()
            # 构建带步数的检查点文件名
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            # 保存模型状态字典
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """
    初始化模型和分词器

    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 加载预训练模型权重
    4. 设置多GPU并行训练（如果可用）
    5. 将模型移动到指定设备
    6. 统计并打印模型参数量

    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    def count_parameters(model):
        """
        统计模型中可训练参数的数量

        Args:
            model: PyTorch模型

        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ==================== 加载分词器 ====================
    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # ==================== 初始化模型 ====================
    # 根据配置创建Transformer模型
    model = Transformer(lm_config)

    # ==================== 加载预训练权重 ====================
    # 加载预训练模型的检查点
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)

    # 移除可能存在的多余前缀（来自某些训练框架）
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # 加载权重到模型（strict=False允许部分权重不匹配）
    model.load_state_dict(state_dict, strict=False)

    # ==================== 多GPU初始化 ====================
    # 检查可用GPU数量并设置DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # 使用DataParallel包装模型以支持多GPU训练
        model = torch.nn.DataParallel(model)

    # ==================== 设备转移 ====================
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(args.device)

    # ==================== 打印模型信息 ====================
    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')

    return model, tokenizer


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Supervised Fine-Tuning")

    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="sft_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型（float32/float16/bfloat16）")

    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="训练数据路径")

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")

    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 多GPU训练参数
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使用的GPU ID，用逗号分隔 (例如: '0,1,2')")

    args = parser.parse_args()

    # ==================== GPU环境设置 ====================
    # 设置可见的GPU设备
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个可用GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ==================== 实验跟踪初始化 ====================
    if args.use_swanlab:
        # 注意：使用前需要先登录 swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # 项目名称
            experiment_name="SFT-215M",  # 实验名称
            config=args,  # 保存所有超参数
        )

    # ==================== 模型配置 ====================
    # 定义语言模型的配置参数
    lm_config = ModelConfig(
        dim=1024,      # 模型维度
        n_layers=18,   # Transformer层数
    )

    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存目录

    # 创建必要的目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 确定设备类型（用于选择合适的上下文管理器）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # ==================== 混合精度训练设置 ====================
    # 设置混合精度训练的上下文管理器
    # CPU训练时使用nullcontext，GPU训练时使用autocast
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器（会加载预训练权重）
    model, tokenizer = init_model()

    # 创建训练数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # 批次大小
        pin_memory=True,             # 将数据加载到固定内存中，加速GPU传输
        drop_last=False,             # 不丢弃最后一个不完整的批次
        shuffle=True,                # 随机打乱数据
        num_workers=args.num_workers # 数据加载的并行工作进程数
    )

    # ==================== 优化器和训练组件初始化 ====================
    # 初始化混合精度训练的梯度缩放器
    # 只有在使用float16或bfloat16时才启用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 初始化AdamW优化器（带权重衰减的Adam）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ==================== 开始训练 ====================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)

    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch)