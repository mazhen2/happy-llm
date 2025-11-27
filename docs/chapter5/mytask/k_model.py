"""
Tiny-K 模型实现

这是一个基于Transformer架构的语言模型，采用了以下关键技术：
1. RMSNorm：根均方归一化，替代LayerNorm
2. RoPE：旋转位置编码，为序列添加位置信息
3. GQA：分组查询注意力，减少KV缓存
4. SwiGLU：门控线性单元激活函数
5. 权重共享：词嵌入层和输出层共享权重

模型架构参考LLaMA设计，但进行了简化优化。
"""

import math
import os
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """
    模型配置类

    继承自PretrainedConfig，用于定义模型的所有超参数。
    这些参数控制模型的规模、结构和行为。
    """
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,  # 模型隐藏层维度，决定模型的容量
            n_layers: int = 12,  # Transformer层数，层数越多模型越深
            n_heads: int = 16,  # 注意力头数，用于多头注意力机制
            n_kv_heads: int = 8,  # 键值对头数，用于GQA（分组查询注意力），通常小于n_heads
            vocab_size: int = 6144,  # 词汇表大小，即token的数量
            hidden_dim: int = None,  # MLP隐藏层维度，如果为None则自动计算
            multiple_of: int = 64,  # 隐藏层维度必须是该值的倍数，用于优化计算效率
            norm_eps: float = 1e-5,  # 归一化层的epsilon值，防止除零
            max_seq_len: int = 512,  # 最大序列长度，超过此长度会被截断
            dropout: float = 0.0,  # Dropout概率，用于防止过拟合
            flash_attn: bool = True,  # 是否使用Flash Attention加速（需要PyTorch >= 2.0）
            **kwargs,
    ):
        """
        初始化模型配置

        Args:
            各参数含义见上方注释
        """
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    将频率张量重塑为可以与输入张量进行广播的形状

    此函数用于将预计算的频率矩阵（freqs_cis）调整为与输入张量x兼容的形状，
    以便在应用旋转位置编码时能够正确进行广播操作。

    广播规则：
        - freqs_cis的原始形状为 (seq_len, head_dim//2)
        - x的形状通常为 (batch_size, seq_len, n_heads, head_dim)
        - 需要将freqs_cis重塑为 (1, seq_len, 1, head_dim//2) 以匹配x的形状

    Args:
        freqs_cis: 频率张量，形状为 (seq_len, head_dim//2)
        x: 输入张量，形状为 (batch_size, seq_len, ..., head_dim)

    Returns:
        重塑后的频率张量，形状为 (1, seq_len, 1, ..., head_dim//2)
    """
    # 获取x的维度数
    ndim = x.ndim
    # 断言，确保维度数至少为2（需要序列维度和特征维度）
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维（序列维度）和最后一维（特征维度）匹配
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 构造新的形状：除了第二维（序列维度）和最后一维（特征维度），其他维度都设为1
    # 这样可以在batch和head维度上进行广播
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码（RoPE）到查询和键向量

    旋转位置编码通过复数旋转的方式将位置信息编码到向量中。
    对于维度为d的向量，将其视为d/2个复数对，每个复数对进行旋转。

    旋转公式（对于复数 z = a + bi）：
        z' = z * e^(iθ) = (a + bi) * (cos(θ) + i*sin(θ))
        = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))

    实部：a' = a*cos(θ) - b*sin(θ)
    虚部：b' = a*sin(θ) + b*cos(θ)

    Args:
        xq: 查询向量，形状为 (batch_size, seq_len, n_heads, head_dim)
        xk: 键向量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        freqs_cos: 余弦频率矩阵，形状为 (seq_len, head_dim//2)
        freqs_sin: 正弦频率矩阵，形状为 (seq_len, head_dim//2)

    Returns:
        xq_out: 应用旋转后的查询向量，形状与xq相同
        xk_out: 应用旋转后的键向量，形状与xk相同
    """
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    # reshape(..., -1, 2) 将最后一个维度分成两半，每两个连续元素组成一个复数对
    # unbind(-1) 将最后一维分离，得到实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播，使其形状与xq_r和xk_r兼容
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转矩阵变换
    # 对于复数 z = a + bi，旋转后的实部：a' = a*cos(θ) - b*sin(θ)
    # 旋转后的虚部：b' = a*sin(θ) + b*cos(θ)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # 查询向量的实部
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # 查询向量的虚部
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # 键向量的实部
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # 键向量的虚部

    # 将实部和虚部重新组合，还原为原始张量的形状
    # stack 将实部和虚部堆叠在一起，flatten(3) 将最后两个维度展平
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    # 将数据类型转换回原始类型（可能是half或bfloat16）
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值对头，用于实现分组查询注意力（GQA）

    在GQA中，查询头数（n_heads）通常大于键值头数（n_kv_heads）。
    为了匹配查询头的数量，需要将每个键值头重复 n_rep = n_heads // n_kv_heads 次。

    例如：
        - n_heads = 8, n_kv_heads = 2, n_rep = 4
        - 每个KV头会被重复4次，使得KV头的总数等于Q头的数量

    Args:
        x: 键或值张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: 重复次数，等于 n_heads // n_kv_heads

    Returns:
        重复后的张量，形状为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    """
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    # 对张量进行扩展和重塑操作以重复键值对
    # 步骤1：在第四个维度（头的维度前）添加一个新的维度
    #       形状从 (bs, slen, n_kv_heads, head_dim) 变为 (bs, slen, n_kv_heads, 1, head_dim)
    # 步骤2：将新添加的维度扩展到n_rep大小，实现重复的效果
    #       形状变为 (bs, slen, n_kv_heads, n_rep, head_dim)
    # 步骤3：重新塑形，合并键/值对头的数量和重复次数的维度
    #       形状变为 (bs, slen, n_kv_heads * n_rep, head_dim)
    return (
        x[:, :, :, None, :]  # 添加新维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展维度
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重塑形状
    )


class Attention(nn.Module):
    """
    多头注意力机制（Multi-head Attention）,支持分组查询注意力（GQA）

    本实现包含以下特性：
    1. 分组查询注意力（GQA）；查询头数可以大于键值头数，减少KV缓存
    2. 旋转位置编码（RoPE）: 通过旋转矩阵编码位置信息
    3. Flash Attention支持：使用PyTorch 2.0+的优化注意力实现
    4. 因果掩码：确保模型只能看到当前位置的信息

    注意力计算公式：
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    其中：
        - Q: 查询矩阵，形状为 (batch, n_heads, seq_len, head_dim)
        - K: 键矩阵，形状为 (batch, n_kv_heads, seq_len, head_dim)
        - V: 值矩阵，形状为 (batch, n_kv_heads, seq_len, head_dim)
    """

    def __init__(self, args: ModelConfig):
        """
        初始化注意力层

        Args:
            args: 模型配置对象，包含所有超参数
        """
        super().__init__()
        # 根据是否指定n_kv_heads,确定键(key)和值(value)的头的数量
        # 若未指定，则使用与查询头相同的数量(标准多头注意力)
        # 托指定且小于n_heads,则使用GQA(分组查询注意力)

        # GQA（Grouped Query Attention）是一种多头注意力的变体，用来减少键和值的计算和显存成本。
        # 传统多头注意力：每个查询头都有对应的键和值头，头数相同。
        # GQA：查询头仍保持较多，但键和值的头数被减少，让多个查询头共享同一组键 / 值。这样在计算 K、V 时只需较少的矩阵乘法，也减少了缓存 / 显存。
        # 之所以可行，是因为 K、V 的表示通常具有较强的冗余性——在不同头之间，键和值往往编码的是类似的上下文信息。

        # 普通 MHA: 8个查询头 → 8个键头 / 8个值头
        # GQA: 8个查询头 → 2个键头 / 2个值头（共享）
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # 确保总头数可以被键值头数整除，这样才能正确实现GQA
        # 例如：n_heads=8, n_kv_heads=2，则每个KV头对应4个Q头
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1（单GPU训练）
        # 在多GPU训练时，可以设置为GPU数量，将注意力头分配到不同GPU
        model_parallel_size = 1

        # 本地计算头数，等于总头数除以模型并行处理大小
        # 在单GPU情况下，等于总头数
        self.n_local_heads = args.n_heads // model_parallel_size

        # 本地键值头数，等于键值头数除以模型并行处理大小
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size

        # 重复次数，用于扩展键和值的尺寸以匹配查询头的数量
        # 例如：n_local_heads=8, n_local_kv_heads=2, 则n_rep=4
        # 每个KV头会被重复4次
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 每个头的维度，等于模型维度除以头的总数
        # 确保总维度 = n_heads * head_dim = dim
        self.head_dim = args.dim // args.n_heads

        # 定义查询（Query）投影矩阵，将输入从dim维度映射到n_heads个head_dim维度
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        # 定义键（Key）投影矩阵，将输入从dim维度映射到n_kv_heads个head_dim维度
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 定义值（Value）投影矩阵，将输入从dim维度映射到n_kv_heads个head_dim维度
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 输出投影矩阵，将所有注意力头的输出合并并映射回dim维度
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 注意力分数dropout,在注意力权重后应用，防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)

        # 参差连接后的dropout,在输出投影后应用
        self.resid_dropout = nn.Dropout(args.dropout)

        # 保存 dropout概率，用于Flash Attention
        # nn.Dropout()是一个函数式实现，它需要你以数值参数的形式把 dropout_p 传进去，而不是直接用模块里的 nn.Dropout 对象
        self.dropout = args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）
        # Flash Attention通过分块计算和在线softmax优化，大幅减少内存占用和计算时间
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

            # 创建因果掩码（上三角矩阵），用于遮蔽未来信息
            # 形状为 (1, 1, max_seq_len, max_seq_len)
            # 上三角部分为-inf，下三角和主对角线为0
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 保留上三角部分（不包括主对角线）

            # 注册为模型的缓冲区，这样会被包含在state_dict中，但不会被视为可训练参数
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """
        前向传播

        计算流程：
        1. 通过线性投影得到 Q,K,V
        2. 应用旋转位置编码（RoPE）
        3. 重复KV以匹配Q头数量
        4. 计算注意力分数并应用softmax
        5. 加权求和得到输出
        6. 通过输出投影层

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            freqs_cos: 余弦频率矩阵，形状为 (seq_len, head_dim//2)
            freqs_sin: 正弦频率矩阵，形状为 (seq_len, head_dim//2)

        head_dim//2:
        freqs_cos（以及配套的 freqs_sin）是给 RoPE（旋转位置编码）用的。
        RoPE 会把每个注意力头的 hidden dimension 拆成一对一对的二维向量来做旋转，每一对共用一个频率（对应正弦、余弦一对值）。
        因此需要把 head_dim 除以 2，得到“有多少对二维向量”，也就是需要多少个余弦/正弦频率。举例：
        如果 head_dim = 128，那就有 128 / 2 = 64 对二维向量；

        Returns:
            注意力输出，形状为 (batch_size, seq_len, dim)
        """

        # 获取批次大小和序列长度
        # x的形状：[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 步骤1：通过线性投影计算查询（Q）、键（K）、值（V）
        # Q: (bsz, seqlen, n_heads * head_dim)
        # K: (bsz, seqlen, n_kv_heads * head_dim)
        # V: (bsz, seqlen, n_kv_heads * head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 步骤2：调整形状以适应多头注意力的结构
        # 将最后一个维度分割成多个头
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 步骤3：应用旋转位置嵌入（RoPE）到查询和键向量
        # RoPE将位置信息编码到向量中，使得注意力能够感知相对位置
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 步骤4：对键和值进行扩展以适应重复次数（GQA）
        # 如果n_kv_heads < n_heads，需要重复KV头以匹配Q头的数量
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 步骤5：将头维度移到批次维度之后，便于批量计算注意力
        # 从 (bsz, seqlen, n_heads, head_dim) 变为 (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 步骤6：计算注意力
        if self.flash:
            # 使用Flash Attention（PyTorch 2.0+）
            # Flash Attention通过分块计算和在线softmax优化，减少内存占用
            # is_causal=True 自动应用因果掩码
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,  # 在训练模式才按配置的概率做 dropout，推理/评估时则设为 0，避免随机性
                is_causal=True
            )
        else:
            # 手动实现注意力机制（兼容旧版本PyTorch）
            # 步骤6.1：计算注意力分数 QK^T / sqrt(d_k)
            # xq: (bsz, n_heads, seqlen, head_dim)
            # xk: (bsz, n_kv_heads, seqlen, head_dim) -> transpose后 (bsz, n_kv_heads, head_dim, seqlen)
            # scores: (bsz, n_heads, seqlen, seqlen)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

            # 步骤6.2：应用因果掩码，遮蔽未来信息
            # 将上三角部分设为-inf，softmax后这些位置的权重为0
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]

            # 步骤6.3：应用softmax得到注意力权重
            # 使用float类型计算以提高数值稳定性，然后转回原类型
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            # 步骤6.4：应用dropout（仅在训练时）
            scores = self.attn_dropout(scores)

            # 步骤6.5：加权求和，得到注意力输出
            # scores: (bsz, n_heads, seqlen, seqlen)
            # xv: (bsz, n_kv_heads, seqlen, head_dim)
            # output: (bsz, n_heads, seqlen, head_dim)
            output = torch.matmul(scores, xv)

        # 步骤7：恢复原始维度顺序并合并所有头
        # 从 (bsz, n_heads, seqlen, head_dim) 变为 (bsz, seqlen, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 步骤8：通过输出投影层，将所有头的输出合并并映射回原始维度
        output = self.wo(output)

        # 步骤9：应用残差dropout（仅在训练时）
        output = self.resid_dropout(output)

        return output


class MLP(nn.Module):
    """
    多层感知机制（MLP）,使用SwiGlu激活函数
    SwiGLU (Swish-Gated Linear Unit) 是GLU的变体，结合了Swish激活函数和门控机制。

    公式：
        SwiGLU(x) = Swish(W1(x)) ⊙ W3(x)
        output = W2(SwiGLU(x))

    其中：
        - Swish(x) = x * sigmoid(x)
        - ⊙ 表示逐元素乘法（Hadamard积）
        - W1, W2, W3 是线性变换层

    优点：
        1. 门控机制允许模型学习更复杂的非线性变换
        2. Swish激活函数平滑且可导，训练更稳定
        3. 相比标准ReLU，表达能力更强
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        """
        初始化MLP层

        Args:
            dim: 输入和输出的维度
            hidden_dim: 隐藏层维度，如果为None则自动计算
            multiple_of: 隐藏层维度必须是该值的倍数（用于优化） 为了计算效率——GPU/TPU 上的矩阵乘法在特定的块大小（比如 64、128）整除时速度更快
            dropout: Dropout概率
        """
        super().__init__()
        # 若没有指定隐藏层维度，按LLaMA的方式自动计算
        if hidden_dim is None:
            # 首先设置为输入维度的4倍（标准Transformer的做法）
            hidden_dim = 4 * dim
            # 然后减少到2/3（LLaMA的优化）
            hidden_dim = int(2 * hidden_dim / 3)
            # 最后确保它是multiple_of的倍数，便于硬件优化（如GPU对齐）
            # 使用向上取整的方式：((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 第一层线性变换：从输入维度到隐藏维度
        # 用于Swish激活函数的输入
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        # 第二层线性变换：从隐藏维度回到输入维度
        # 用于输出投影
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        # 第三层线性变换：从输入维度到隐藏维度
        # 用于门控（gate）机制
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        计算流程：
        1. x通过w1得到h1，应用Swish激活函数
        2. x通过w3得到h3（门控信号）
        3. h1和h3逐元素相乘（门控机制）
        4. 结果通过w2投影回原始维度
        5. 应用dropout

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)

        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        # Swish激活函数：Swish(x) = x * sigmoid(x)
        # self.w1(x) 把输入 x 映射到更高维的隐层空间，然后用 F.silu（PyTorch 里的 Swish）做非线性激活：
        # h1 = Swish(w1(x)) = w1(x) * sigmoid(w1(x))

        # 门控机制：将Swish(w1(x))与w3(x)逐元素相乘
        # self.w3(x) 也把 x 映射到同样的隐层维度，但不做激活，直接作为门控向量 h3。

        # h1 * h3 就是 SwiGLU 的核心：Swish(w1(x)) ⊙ w3(x)。
        # 门控向量决定每个隐层通道的开放程度，可以理解为为每个特征通道加了一扇“门”，让网络学到更细粒度的控制。

        # eg.
        # # 对于每个位置 (i, j, k)：output[i, j, k] = h1[i, j, k] * h3[i, j, k]
        # h3 的每个元素作为一个“开关系数”，控制 h1 对应位置的信息通过量：
        # h3[k] ≈ 0：关闭，h1[k] 的信息几乎被抑制
        # h3[k] ≈ 1：全开，h1[k] 的信息几乎全部通过
        # h3[k] > 1：放大，h1[k] 的信息被增强
        # h3[k] < 0：反向，h1[k] 的信息被反转

        # 最后通过w2投影并应用dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)

    RMSNorm是LayerNorm的简化版本，只对输入进行归一化而不减去均值。
    相比LayerNorm，RMSNorm计算更简单，但效果相当。

    数学公式：
        RMS(x) = sqrt(mean(x^2))
        RMSNorm(x) = (x / RMS(x)) * weight

    优点：
        1. 计算效率更高（不需要计算均值）
        2. 数值稳定性更好
        3. 在大多数任务上效果与LayerNorm相当
    """

    def __init__(self, dim: int, eps: float):
        """
        初始化RMSNorm层

        Args:
            dim: 输入特征的维度
            eps: 防止除零的小常数，通常设为1e-5或1e-6
        """
        super().__init__()
        # eps是为了防止除以0的情况，当输入全为0时避免数值不稳定
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        # 用于在归一化后对特征进行缩放，允许模型学习合适的特征尺度
        # nn.Parameter() 将张量注册为模型参数，会被优化器更新
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        计算RMSNorm的核心归一化操作

        步骤：
        1. 计算输入x在最后一个维度上的平方均值：mean(x^2)
        2. 计算平方根的倒数：1/sqrt(mean(x^2) + eps)
        3. 将输入x乘以归一化因子

        Args:
            x: 输入张量，形状为 (..., dim)

        Returns:
            归一化后的张量，形状与输入相同
        """
        # x.pow(2).mean(-1, keepdim=True) 计算输入x在最后一个维度上的平方均值
        # torch.rsqrt 是平方根的倒数（1/sqrt），比先sqrt再除更高效
        # 加上eps防止分母为0，保证数值稳定性
        # 最后乘以x，得到归一化后的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，可以是任意形状，但最后一个维度必须是dim

        Returns:
            归一化并缩放后的张量，形状与输入相同
        """
        # 首先将输入x转为float类型进行计算，提高数值精度
        # 然后进行RMSNorm归一化
        # 最后转回原来的数据类型（可能是half或bfloat16）
        # 乘以可学习的weight参数，允许模型调整特征尺度
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class DecoderLayer(nn.Module):
    """
    Transformer 解码器
    每个解码器包含：
    1. 预归一化的多头注意力（Pre-norm Attention）
    2. 残差连接（Residual Connection）
    3. 预归一化的前馈网络（Pre-norm Feed Forward）
    4. 残差连接（Residual Connection）

    架构（Pre-norm）：
        h = x + Attention(RMSNorm(x))
        out = h + MLP(RMSNorm(h))
        
    执行顺序：
        第1步：h = x + Attention(RMSNorm(x))
        RMSNorm(x)：先归一化输入 x
        Attention(...)：对归一化后的结果做注意力
        x + Attention(...)：残差连接（原始输入 + 注意力输出）
        第2步：out = h + MLP(RMSNorm(h))
        RMSNorm(h)：归一化 h
        MLP(...)：对归一化后的结果做前馈
        h + MLP(...)：残差连接（h + 前馈输出）

    注意：使用Pre-norm而非Post-norm，训练更稳定
    """

    def __init__(self, layer_id: int, args: ModelConfig):
        """
        初始化解码器层
        Args:
            layer_id: 层的索引，用于标识不同层
            args: 模型配置对象
        """
        super().__init__()
        # 保存配置参数
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # 多头注意力层，支持GQA和RoPE
        self.attention = Attention(args)

        # 前馈网络层，使用SwiGLU激活函数
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        # 层的ID，可用于层特定的操作（如不同的学习率）
        self.layer_id = layer_id

        # 注意力层的归一化(Pre-norm)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 前馈神经网络的归一化(Pre-norm)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        """
        前向传播
        
        计算流程（Pre-norm架构）
            1. 对输入x进行RMSNorm归一化
            2. 通过注意力层，得到注意力输出
            3. 将注意力输出与原始输入相加（残差连接）
            4. 对输出h进行RMSNorm归一化
            5. 通过前馈神经网络，得到前馈输出
            6. 将前馈输出与h相加（残差连接）
        
        Args:
            x: 输入张量，形状为（batch_size,seq_len,dim）
            freqs_cos: 余弦频率矩阵，用于RoPE
            freqs_sin: 正弦频率矩阵，用于RoPE
        Returns:
            输出张量，形状为（batch_size,seq_len,dim）
        """

        # 第一部分，注意力+残差连接
        # Pre-norm: 先归一化，在计算注意力，最后残差连接
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)

        # 第二部分，前馈神经网络+残差连接
        # Pre-norm: 先归一化，在计算前馈，最后残差连接
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码（RoPE）的频率矩阵

    RoPE (Rotary Position Embedding) 是一种相对位置编码方法，
    通过旋转矩阵将位置信息编码到查询和键向量中。

    核心思想：
        对于位置m的向量x，通过旋转矩阵R_theta^m将其旋转，
        使得不同位置的向量之间的内积能够反映相对位置关系。

    数学原理：
        1. 计算频率：freq_i = 1 / (theta^(2i/dim))，i从0到dim/2-1
        2. 对于位置pos，计算角度：angle = pos * freq
        3. 使用cos和sin生成旋转矩阵的系数

    注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入

    Args:
        dim: 每个注意力头的维度（head_dim），必须是偶数
        end: 最大序列长度，用于预计算所有位置
        theta: 频率基数，控制频率的衰减速度，默认10000.0

    Returns:
        freqs_cos: 余弦频率矩阵，形状为 (end, dim//2)
        freqs_sin: 正弦频率矩阵，形状为 (end, dim//2)
    """
    # 生成频率序列：对于维度i，频率为 1/(theta^(2i/dim))
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ..., dim-2]
    # [: (dim // 2)] 确保只取前dim/2个元素
    # 每个元素除以dim，再作为theta的指数，最后取倒数得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置序列：从0到end-1的所有位置
    t = torch.arange(end, device=freqs.device)

    # 计算外积：将位置t与频率freqs相乘
    # 结果矩阵的每一行对应一个位置，每一列对应一个频率维度
    # 形状：(end, dim//2)
    freqs = torch.outer(t, freqs).float()

    # 计算余弦值，作为旋转矩阵的实部系数
    freqs_cos = torch.cos(freqs)

    # 计算正弦值，作为旋转矩阵的虚部系数
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


class Transformer(PreTrainedModel):
    """
    Tiny-K Transformer模型

    完整的Transformer解码器架构，包含：
    1. 词嵌入层（Token Embedding）
    2. 多层解码器（Decoder Layers）
    3. 输出层（Output Layer）
    4. 权重共享（Embedding和Output共享权重）
    5. 旋转位置编码（RoPE）

    模型流程：
        tokens -> Embedding -> Dropout -> DecoderLayers -> RMSNorm -> Output -> logits
    """

    config_class = ModelConfig  # 配置类，用于保存和加载模型配置
    last_loss: Optional[torch.Tensor]  # 记录最后一次计算的损失，用于调试

    def __init__(self, args: ModelConfig = None):
        """
        初始化 Transformer 模型
        Args:
            args: 模型配置参数，包含所有超参数
        """
        super().__init__(args)
        # 保存配置参数
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 词嵌入层：将token ID映射为向量
        # 输入：token IDS (batch_size,seq_len)
        # 输出：词向量 (batch_size,seq_len,dim)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # 输入 dropout 层，在词嵌入后应用，防止过拟合
        self.dropout = nn.Dropout(args.dropout)

        # 创建多层解码器
        # 使用ModuleList存储所有层，便于索引和便利
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # 最终归一化层，在所有解码器层之后应用
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 输出层:将隐藏状态映射回词汇表空间
        # 输出logits,形状为  (batch_size, seq_len, vocab_size)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 权重共享：将词嵌入层的权重与输出层的权重共享
        # 优点：
        # 1. 减少参数量（节省vocab_size * dim 个参数）、
        # 2. 减少训练量
        # 3. 在语言模型中通常能提升性能
        self.tok_embeddings.weight = self.output.weight

        # 预计算旋转位置编码的频率矩阵
        # 在初始化时计算，避免每次前向传播重复计算
        # dim参数是每个头的维度：dim // n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
        )
        # 注册为缓冲区，会被包含在state_dict中，但不会被视为可训练参数
        # persistent=False表示不会保存到checkpoint（可以重新计算）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        # 对每个模块递归调用_init_weights方法
        self.apply(self._init_weights)

        # 对残差投影进行特殊的缩放初始化
        # 这是LLaMA的初始化策略，有助于深层网络的训练稳定性
        # 缩放因子：1/sqrt(2*n_layers)，随着层数增加，初始化方差减小
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * args.n_layers)
                )

        # 初始化输出相关的属性
        self.last_loss = None  # 最后一次计算的损失
        self.OUT = CausalLMOutputWithPast()  # 输出容器，用于返回logits和loss
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表（用于模型并行）

    def _init_weights(self, module):
        """
        权重初始化函数

        使用Xavier初始化的变体（正态分布）
        - 线性层：均值为0，标准差为0.02的正态分布
        - 嵌入层：均值为0，标准差为0.02的正态分布
        - 偏置：初始化为0

        Args:
            module: 要初始化的模块（Linear或Embedding）
        """
        if isinstance(module, nn.Linear):
            # 线性层权重初始化：正态分布，标准差0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果存在偏置，初始化为0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层权重初始化：正态分布，标准差0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播

        完整的模型前向传播流程：
        1. Token嵌入：将token IDs转换为向量
        2. Dropout：防止过拟合
        3. 多层解码器：通过所有Transformer层
        4. 最终归一化：RMSNorm
        5. 输出投影：映射到词汇表空间
        6. 计算损失（如果提供了targets）

        Args:
            tokens: 输入token张量，形状为 (batch_size, seq_len)
            targets: 目标token张量（用于训练），形状为 (batch_size, seq_len)
            **kwargs: 其他关键字参数，支持input_ids和attention_mask（兼容transformers接口）

        Returns:
            CausalLMOutputWithPast对象，包含：
            - logits: 模型输出logits，形状为 (batch_size, seq_len, vocab_size)
            - last_loss: 损失值（如果提供了targets）
        """
        # 兼容transformers库的接口
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # 获取批次大小和序列长度
        _bsz, seqlen = tokens.shape

        # 步骤1：通过词嵌入层，将token IDs转换为向量
        # 输入：(batch_size, seq_len)
        # 输出：(batch_size, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # 步骤2：应用dropout（仅在训练时）
        h = self.dropout(h)

        # 步骤3：获取当前序列长度对应的旋转位置编码频率
        # 只取前seqlen个位置的频率（如果序列长度小于max_seq_len）
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 步骤4：通过所有解码器层
        # 每一层都会应用注意力机制和前馈网络
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # 步骤5：应用最终归一化层
        h = self.norm(h)

        # 步骤6：计算输出和损失
        if targets is not None:
            # 训练模式：计算所有位置的logits和损失
            logits = self.output(h)  # (batch_size, seq_len, vocab_size)

            # 计算交叉熵损失
            # 将logits和targets展平为2D张量
            # ignore_index=0表示忽略padding token（ID为0）
            # reduction='none'返回每个样本的损失，不进行平均
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction='none'
            )
        else:
            # 推理模式：只计算最后一个位置的logits（优化）
            # 在自回归生成时，只需要最后一个位置的输出
            logits = self.output(h[:, [-1], :])  # (batch_size, 1, vocab_size)
            self.last_loss = None

        # 设置输出容器
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]  # 只返回生成的token

    def _greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        贪婪解码：选择概率最大的token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        return idx_next

    def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        随机采样：基于概率分布随机选择token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)
            temperature: 温度参数，控制随机性
            top_k: 只考虑概率最高的k个token

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        # 缩放 logits
        logits = logits / temperature

        # 应用top-k过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将不在 top-k 内的 logits 设为负无穷
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int,
                     temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
        """
        束搜索：维护多个候选序列，选择最优路径

        束搜索的核心思想：在每一步生成时，不是只选择一个最佳token，
        而是保留多个候选路径，最终选择累积概率最高的完整序列。

        Args:
            idx: 输入序列，形状为 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            num_beams: 束宽度，表示保留的候选路径数量
            temperature: 温度参数，控制分布的平滑程度
            top_k: top-k过滤参数，限制候选token范围
            stop_id: 停止生成的token ID，遇到则停止

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
            只返回新生成的部分，不包含原始输入序列
        """
        # 获取输入序列的基本信息
        batch_size = idx.shape[0]  # 批次大小，通常为1
        seq_len = idx.shape[1]  # 输入序列长度

        # 初始化束：创建 num_beams 个候选序列
        beams = [idx.clone() for _ in range(num_beams)]
        # 初始化每个候选序列的累积对数概率分数
        beam_scores = torch.zeros(num_beams, device=idx.device)
        # 第一个候选是原始输入序列，分数为0
        beam_scores[0] = 0.0
        # 其他候选初始分数设为负无穷，表示尚未生成
        beam_scores[1:] = float('-inf')

        # 主循环：逐步生成新的token，最多生成 max_new_tokens 个
        for step in range(max_new_tokens):
            # 每轮迭代收集新的候选序列和分数
            new_beams = []  # 新的候选序列列表
            new_scores = []  # 对应的分数列表

            # 遍历当前的所有候选序列
            for beam_idx, beam in enumerate(beams):
                # 跳过无效候选（分数为负无穷的序列）
                if beam_scores[beam_idx] == float('-inf'):
                    continue

                # 序列长度检查：如果超过最大长度，截取最后的部分
                beam_cond = beam if beam.size(1) <= self.args.max_seq_len else beam[:, -self.args.max_seq_len:]

                # 前向传播：获取模型对当前序列的预测
                output = self(beam_cond)
                # 提取最后一个位置的logits，用于预测下一个token
                logits = output.logits[:, -1, :]  # 形状: (1, vocab_size)

                # 温度缩放：调整logits的分布
                if temperature != 1.0:
                    logits = logits / temperature
                    # 温度 > 1：分布更平滑，增加随机性
                    # 温度 < 1：分布更尖锐，更确定

                # Top-k过滤：限制候选token的范围，提高质量
                if top_k is not None:
                    # 找到logits中前top_k个最大的值
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 将不在前top_k内的logits设为负无穷
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    # 这样采样时只会考虑前top_k个token

                # 计算对数概率：使用log_softmax避免数值不稳定
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取前 num_beams 个最可能的候选token
                # 注意：这里的top-k与上面的top-k不同
                # 上面的top-k是全局过滤，这里是束搜索的分支选择
                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                # 为当前候选序列生成 num_beams 个扩展序列
                for k in range(num_beams):
                    # 选择第k个候选token
                    token = top_indices[:, k:k + 1]  # token ID
                    log_prob = top_log_probs[:, k]  # 对应的对数概率

                    # 扩展序列：将新token添加到当前序列末尾
                    new_beam = torch.cat([beam, token], dim=1)
                    # 更新累积分数：原序列分数 + 新token的对数概率
                    new_score = beam_scores[beam_idx] + log_prob.item()

                    # 保存新的候选序列和分数
                    new_beams.append(new_beam)
                    new_scores.append(new_score)

            # 安全检查：如果没有生成任何有效候选，提前结束
            if not new_beams:
                break

            # 筛选最佳候选：从所有新生成的候选中选择分数最高的 num_beams 个
            # 按分数降序排序，获取索引
            sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
            # 选择前 num_beams 个最佳候选
            beams = [new_beams[i] for i in sorted_indices[:num_beams]]
            beam_scores = [new_scores[i] for i in sorted_indices[:num_beams]]

            # 停止条件检查：检查最佳序列是否以停止token结尾
            if stop_id is not None and beams[0][0, -1] == stop_id:
                break

        # 返回得分最高的序列，只返回新生成的部分（去掉原始输入）
        # beams[0] 是最终得分最高的完整序列
        # [:, seq_len:] 切片只保留生成部分
        return beams[0][:, seq_len:]

    @torch.inference_mode()
    def generate_super(self,
                       idx,
                       stop_id=None,
                       max_new_tokens=256,
                       temperature=1.0,
                       top_k=None,
                       do_sample=False,
                       num_beams=1
                       ):
        """
        高级文本生成函数，支持三种解码策略：

        1. 贪婪解码（Greedy Search）：
           - 参数：do_sample=False, num_beams=1
           - 特点：每步选择概率最大的token，速度快、结果确定

        2. 随机采样（Random Sampling）：
           - 参数：do_sample=True, num_beams=1
           - 特点：基于概率分布随机采样，可配合temperature和top-k控制多样性

        3. 束搜索（Beam Search）：
           - 参数：do_sample=False, num_beams>1
           - 特点：维护多条候选路径，选择总概率最高的序列，质量更高但速度较慢

        Args:
            idx: 输入序列张量，形状为 (batch_size, seq_len)
            stop_id: 停止生成的token ID
            max_new_tokens: 最大生成token数量
            temperature: 温度参数，控制随机性，越高越随机
            top_k: 只考虑概率最高的k个token，None表示不考虑
            do_sample: 是否使用随机采样，False时使用确定性解码
            num_beams: 束搜索的束宽度，1表示不使用束搜索

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
        """
        # 参数验证
        if temperature <= 0:
            temperature = 0.001  # 避免除零错误
        if num_beams < 1:
            num_beams = 1
        if top_k is not None and top_k < 1:
            top_k = None

        # 束搜索逻辑
        if not do_sample and num_beams > 1:
            return self._beam_search(idx, max_new_tokens, num_beams, temperature, top_k, stop_id)

        # 贪婪解码和随机采样逻辑
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出

            # 根据参数选择解码策略
            if do_sample:
                idx_next = self._random_sample(logits, temperature, top_k)
            else:
                # 当temperature=0时使用贪婪解码
                if temperature < 0.1:
                    idx_next = self._greedy_decode(logits)
                else:
                    # 低温度下的随机采样（接近贪婪）
                    idx_next = self._random_sample(logits, temperature, top_k)

            # 检查停止条件
            if stop_id is not None and idx_next[0, 0] == stop_id:
                break

            # 将选择的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]  # 只返回生成的token


if __name__ == '__main__':
    """
    主函数：模型测试和示例

    演示如何使用Tiny-K模型进行前向传播：
    1. 加载tokenizer
    2. 创建模型配置和实例
    3. 计算模型参数量
    4. 准备输入数据
    5. 进行前向传播
    """
    # 加载tokenizer，用于将文本转换为token IDs
    # 获取当前文件所在目录，然后构建 tokenizer 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir, "../code/tokenizer_k/")
    tokenizer_path = os.path.normpath(tokenizer_path)  # 规范化路径
    # 添加 local_files_only=True 避免从 Hugging Face Hub 下载
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    # 创建模型配置
    # dim=1024: 模型隐藏层维度为1024
    # n_layers=18: 使用18层Transformer
    # 其他参数使用默认值
    args = ModelConfig(
        dim=1024,
        n_layers=18,
    )

    # 实例化Transformer模型
    model = Transformer(args=args)

    # 计算模型的总参数量
    # numel()返回张量中元素的总数
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')

    # 准备测试文本
    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    # 添加开始和结束标记
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    # 将文本转换为token IDs
    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    # 验证tokenization：将token IDs转换回文本
    print("dcode_str :", tokenizer.decode(input_id))

    # 准备训练数据
    # X: 输入序列（去掉最后一个token）
    # Y: 目标序列（去掉第一个token，用于下一个token预测）
    # unsqueeze(0)添加batch维度，形状从(seq_len,)变为(1, seq_len)
    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # 将输入张量传入模型进行前向传播
    # 模型会计算logits和损失
    output = model(X, Y)
