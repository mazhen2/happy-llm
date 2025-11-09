"""
Transformer 完整实现
包含编码器（Encoder）和解码器（Decoder）的完整 Transformer 模型
基于 "Attention is All You Need" 论文实现
"""

import torch
import math
from torch import nn
from dataclasses import dataclass
from transformers import BertTokenizer
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    模型配置参数类

    Attributes:
        n_embd: 词嵌入维度，即每个 token 被表示为多少维的向量
        n_heads: 多头注意力的头数
        dim: 模型的隐藏层维度（通常等于 n_embd）
        dropout: Dropout 概率，用于防止过拟合
        max_seq_len: 最大序列长度，用于创建位置编码和注意力掩码
        vocab_size: 词表大小，即模型能识别的不同 token 的数量
        block_size: 输入序列的最大长度限制
        n_layer: Encoder 和 Decoder 的层数
    """
    n_embd: int  # 嵌入维度
    n_heads: int  # 头数
    dim: int  # 模型维度
    dropout: float  # Dropout 比例
    max_seq_len: int  # 最大序列长度
    vocab_size: int  # 词表大小
    block_size: int  # 块大小
    n_layer: int  # 层数


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）

    多头注意力允许模型在不同的表示子空间中联合关注来自不同位置的信息。
    通过将查询、键、值投影到多个不同的表示空间，模型可以同时关注不同方面的信息。

    Args:
        args: 模型配置参数
        is_causal: 是否使用因果（causal）注意力掩码。
                  True 表示当前位置只能看到之前的位置（用于 Decoder）
                  False 表示可以看到所有位置（用于 Encoder）
    """

    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入平均拆分成 n_heads 个子空间
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数
        # 例如：dim=512, n_heads=8, 则 head_dim=64
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # 定义 Query, Key, Value 的线性变换矩阵
        # 输入维度：n_embd，输出维度：n_heads * head_dim
        # 这里通过一个大的线性层来代替 n_heads 个小的线性层
        # 原理：(AB)C = A(BC)，先分别变换再拼接 = 拼接后统一变换
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)

        # 输出投影矩阵，将多头的结果投影回原始维度
        # 维度：(n_heads * head_dim) -> dim
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        # 注意力权重的 Dropout，用于防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接前的 Dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 如果是因果注意力（Decoder 中使用），需要创建一个上三角掩码矩阵
        # 掩码矩阵用于防止当前位置看到未来的信息
        if is_causal:
            # 创建形状为 [1, 1, max_seq_len, max_seq_len] 的矩阵，填充为负无穷
            # 维度说明：[batch, n_heads, seq_len, seq_len]
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            # torch.triu 返回上三角矩阵，diagonal=1 表示对角线上方的元素保留
            # 这样在注意力计算时，当前位置无法关注到未来位置（因为被设为 -inf）
            mask = torch.triu(mask, diagonal=1)
            # 将 mask 注册为 buffer，这样它会随模型一起保存和加载，但不会被当作参数更新
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        多头注意力前向传播

        Args:
            q: Query 张量，形状 [batch_size, seq_len, n_embd]
            k: Key 张量，形状 [batch_size, seq_len, n_embd]
            v: Value 张量，形状 [batch_size, seq_len, n_embd]

        Returns:
            output: 注意力输出，形状 [batch_size, seq_len, dim]

        注意：
            - 在自注意力中，q=k=v（来自同一个输入）
            - 在交叉注意力中，q 来自 decoder，k 和 v 来自 encoder
        """

        # 步骤 1: 获取输入的形状信息
        # bsz: batch size（批次大小）
        # seqlen: sequence length（序列长度）
        # 输入形状: [batch_size, seq_len, n_embd]
        bsz, seqlen, _ = q.shape

        # 步骤 2: 线性变换得到 Q, K, V
        # 通过线性层将输入投影到多头空间
        # 变换: (B, T, n_embd) -> (B, T, n_heads * head_dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 步骤 3: 重塑张量以分离多头
        # 将 Q、K、V 从 (B, T, n_heads * head_dim) 重塑为 (B, T, n_heads, head_dim)
        # 然后交换维度得到 (B, n_heads, T, head_dim)
        #
        # 为什么要这样做？
        # - view 操作会按照内存顺序重新组织数据
        # - 先 view 再 transpose 可以确保每个头获得连续的特征维度
        # - 最终得到的形状便于进行批量矩阵乘法
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)  # (B, T, nh, hs)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)  # (B, T, nh, hs)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)  # (B, T, nh, hs)

        # 交换维度：将 n_heads 维度移到第二个位置
        # 从 (B, T, nh, hs) 变为 (B, nh, T, hs)
        xq = xq.transpose(1, 2)  # (B, nh, T, hs)
        xk = xk.transpose(1, 2)  # (B, nh, T, hs)
        xv = xv.transpose(1, 2)  # (B, nh, T, hs)

        # 步骤 4: 计算注意力分数
        # 注意力公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

        # 计算 Q @ K^T，得到注意力分数矩阵
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # 除以 sqrt(head_dim) 进行缩放，防止点积过大导致 softmax 梯度消失
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 步骤 5: 应用因果掩码（如果需要）
        # 在 Decoder 中，需要防止当前位置关注到未来的位置
        if self.is_causal:
            assert hasattr(self, 'mask'), "因果注意力需要掩码矩阵"
            # 截取掩码到当前序列长度（因为实际序列可能比 max_seq_len 短）
            # 加上 mask（-inf）会让对应位置的注意力权重变为 0
            scores = scores + self.mask[:, :, :seqlen, :seqlen]

        # 步骤 6: 计算注意力权重
        # 对最后一维（key 维度）应用 softmax，得到注意力权重
        # 转换为 float 以提高数值稳定性，然后转回原类型
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (B, nh, T, T)

        # 应用 Dropout 到注意力权重
        scores = self.attn_dropout(scores)

        # 步骤 7: 计算加权和
        # 用注意力权重对 Value 进行加权求和
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 步骤 8: 合并多头
        # 将多头的结果拼接回原始维度
        # 先交换维度: (B, nh, T, hs) -> (B, T, nh, hs)
        # 再重塑: (B, T, nh, hs) -> (B, T, nh * hs)
        #
        # contiguous() 的作用：
        # - transpose 不会改变底层内存布局，只是改变了索引方式
        # - view 要求张量在内存中是连续的
        # - contiguous() 会创建一个新的连续内存副本
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 步骤 9: 输出投影
        # 通过输出线性层和 Dropout 得到最终输出
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）

    与 Batch Normalization 不同，Layer Norm 对每个样本的所有特征进行归一化，
    而不是对整个 batch 的同一个特征进行归一化。这使得它更适合序列模型和小 batch。

    公式: LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
    其中 μ 和 σ 是在特征维度上计算的均值和标准差

    Args:
        features: 特征维度大小
        eps: 防止除零的小常数，默认 1e-6
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 可学习的缩放参数 γ（gamma），初始化为 1
        self.a_2 = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数 β（beta），初始化为 0
        self.b_2 = nn.Parameter(torch.zeros(features))
        # epsilon：防止除零的小常数
        self.eps = eps

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, features]

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算最后一个维度（特征维度）的均值
        # keepdim=True 保持维度，便于广播
        # mean: [batch_size, seq_len, 1]
        mean = x.mean(-1, keepdim=True)

        # 计算最后一个维度的标准差
        # std: [batch_size, seq_len, 1]
        std = x.std(-1, keepdim=True)

        # 归一化：(x - μ) / (σ + ε)
        # 然后进行缩放和偏移：γ * normalized + β
        # 这里利用了广播机制，a_2 和 b_2 会自动扩展到匹配 x 的形状
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network / MLP）

    在 Transformer 中，每个 Encoder/Decoder 层都包含一个前馈网络。
    它由两个线性层组成，中间使用 ReLU 激活函数。

    结构: Linear -> ReLU -> Dropout -> Linear -> Dropout

    Args:
        dim: 输入和输出维度
        hidden_dim: 隐藏层维度（通常是 dim 的 4 倍）
        dropout: Dropout 概率
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 第一层线性变换：dim -> hidden_dim（扩展维度）
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 第二层线性变换：hidden_dim -> dim（恢复原始维度）
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Dropout 层，用于正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, dim]

        Returns:
            输出张量，形状 [batch_size, seq_len, dim]
        """
        # x -> Linear(w1) -> ReLU -> Linear(w2) -> Dropout -> output
        # 这个两层结构允许模型学习更复杂的非线性变换
        return self.dropout(self.w2(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """
    Encoder 层

    每个 Encoder 层包含两个子层：
    1. 多头自注意力机制
    2. 前馈神经网络

    每个子层都使用残差连接（Residual Connection）和层归一化（Layer Normalization）
    结构: LayerNorm -> MultiHeadAttention -> 残差连接 -> LayerNorm -> FFN -> 残差连接

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super().__init__()
        # 第一个 LayerNorm，在自注意力之前
        self.attention_norm = LayerNorm(args.n_embd)
        # 多头自注意力层（不使用因果掩码，因为 Encoder 可以看到所有位置）
        self.attention = MultiHeadAttention(args, is_causal=False)
        # 第二个 LayerNorm，在前馈网络之前
        self.fnn_norm = LayerNorm(args.n_embd)
        # 前馈神经网络
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]

        Returns:
            输出张量，形状 [batch_size, seq_len, n_embd]
        """
        # 子层 1: 自注意力 + 残差连接
        # Pre-LN 架构：先做 LayerNorm，再做注意力
        x_norm = self.attention_norm(x)
        # 自注意力：q, k, v 都来自同一个输入
        # 使用残差连接：输出 = 输入 + 注意力结果
        h = x + self.attention.forward(x_norm, x_norm, x_norm)

        # 子层 2: 前馈网络 + 残差连接
        # 同样使用 Pre-LN 架构
        h_norm = self.fnn_norm(h)
        out = h + self.feed_forward.forward(h_norm)
        return out


class Encoder(nn.Module):
    """
    Transformer Encoder

    Encoder 由多个 EncoderLayer 堆叠而成，用于处理输入序列。
    在机器翻译任务中，Encoder 负责编码源语言序列。

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super(Encoder, self).__init__()
        # 创建 N 个 Encoder 层并组成列表
        # nn.ModuleList 确保这些层的参数会被正确注册
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        # 最后的 LayerNorm 层
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        """
        前向传播，依次通过所有 Encoder 层

        Args:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]

        Returns:
            编码后的张量，形状 [batch_size, seq_len, n_embd]
        """
        # 依次通过每个 Encoder 层
        for layer in self.layers:
            x = layer(x)
        # 最后进行一次 LayerNorm
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder 层

    每个 Decoder 层包含三个子层：
    1. 带掩码的多头自注意力（Masked Multi-Head Self-Attention）
    2. 多头交叉注意力（Multi-Head Cross-Attention），关注 Encoder 的输出
    3. 前馈神经网络（Feed-Forward Network）

    每个子层都使用残差连接和层归一化

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super().__init__()
        # 第一个 LayerNorm：在掩码自注意力之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # 掩码自注意力：防止当前位置看到未来的信息（is_causal=True）
        self.mask_attention = MultiHeadAttention(args, is_causal=True)

        # 第二个 LayerNorm：在交叉注意力之前
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # 交叉注意力：Query 来自 Decoder，Key 和 Value 来自 Encoder
        # 不需要掩码（is_causal=False），因为可以关注 Encoder 的所有位置
        self.attention = MultiHeadAttention(args, is_causal=False)

        # 第三个 LayerNorm：在前馈网络之前
        self.ffn_norm = LayerNorm(args.n_embd)
        # 前馈神经网络
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        """
        前向传播

        Args:
            x: Decoder 输入张量，形状 [batch_size, tgt_seq_len, n_embd]
            enc_out: Encoder 输出张量，形状 [batch_size, src_seq_len, n_embd]

        Returns:
            输出张量，形状 [batch_size, tgt_seq_len, n_embd]
        """
        # 子层 1: 掩码自注意力 + 残差连接
        # 这允许 Decoder 每个位置关注到之前的所有位置（但不能看到未来）
        x_norm = self.attention_norm_1(x)
        x = x + self.mask_attention.forward(x_norm, x_norm, x_norm)

        # 子层 2: 交叉注意力 + 残差连接
        # Query 来自 Decoder，Key 和 Value 来自 Encoder 的输出
        # 这允许 Decoder 关注到输入序列的所有位置
        x_norm = self.attention_norm_2(x)
        h = x + self.attention.forward(x_norm, enc_out, enc_out)

        # 子层 3: 前馈网络 + 残差连接
        h_norm = self.ffn_norm(h)
        out = h + self.feed_forward.forward(h_norm)
        return out


class Decoder(nn.Module):
    """
    Transformer Decoder

    Decoder 由多个 DecoderLayer 堆叠而成，用于生成输出序列。
    在机器翻译任务中，Decoder 负责生成目标语言序列。

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super(Decoder, self).__init__()
        # 创建 N 个 Decoder 层并组成列表
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        # 最后的 LayerNorm 层
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        """
        前向传播，依次通过所有 Decoder 层

        Args:
            x: Decoder 输入，形状 [batch_size, tgt_seq_len, n_embd]
            enc_out: Encoder 输出，形状 [batch_size, src_seq_len, n_embd]

        Returns:
            解码后的张量，形状 [batch_size, tgt_seq_len, n_embd]
        """
        # 依次通过每个 Decoder 层
        for layer in self.layers:
            x = layer(x, enc_out)
        # 最后进行一次 LayerNorm
        return self.norm(x)


class PositionalEncoding(nn.Module):
    """
    位置编码（Positional Encoding）

    由于 Transformer 没有 RNN 的顺序处理特性，需要通过位置编码来注入位置信息。
    使用正弦和余弦函数来生成位置编码，具有以下优点：
    1. 可以处理任意长度的序列
    2. 不同位置的相对位置关系可以被模型学习
    3. 不需要训练，是固定的

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中 pos 是位置，i 是维度索引

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵，形状 [block_size, n_embd]
        pe = torch.zeros(args.block_size, args.n_embd)

        # 创建位置索引 [0, 1, 2, ..., block_size-1]，形状 [block_size, 1]
        position = torch.arange(0, args.block_size).unsqueeze(1)

        # 计算除数项 (div_term)
        # 对于偶数维度 i: 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        # 这里使用 exp 和 log 来避免数值溢出
        # torch.arange(0, n_embd, 2) 生成 [0, 2, 4, ..., n_embd-2]
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )

        # 对偶数位置使用 sin，对奇数位置使用 cos
        # pe[:, 0::2] 表示所有行，从第 0 列开始每隔 2 列取一个（即偶数列）
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] 表示所有行，从第 1 列开始每隔 2 列取一个（即奇数列）
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度：[block_size, n_embd] -> [1, block_size, n_embd]
        pe = pe.unsqueeze(0)

        # 将位置编码注册为 buffer
        # buffer 不会被当作模型参数，但会被保存在 state_dict 中
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        将位置编码加到输入上

        Args:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]

        Returns:
            加上位置编码的张量，形状与输入相同
        """
        # 取出对应长度的位置编码，并加到输入上
        # requires_grad_(False) 确保位置编码不参与梯度计算
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    """
    完整的 Transformer 模型

    包含 Encoder 和 Decoder 的完整 Transformer 架构，适用于序列到序列任务。
    模型结构:
        输入 -> Embedding -> 位置编码 -> Encoder -> Decoder -> 输出投影 -> Logits

    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super().__init__()
        # 验证必要参数
        assert args.vocab_size is not None, "必须指定词表大小 (vocab_size)"
        assert args.block_size is not None, "必须指定最大序列长度 (block_size)"
        self.args = args

        # 构建 Transformer 的各个组件
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层：将 token id 转换为向量
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            # 位置编码：添加位置信息
            wpe=PositionalEncoding(args),
            # Dropout 层
            drop=nn.Dropout(args.dropout),
            # Encoder 模块
            encoder=Encoder(args),
            # Decoder 模块
            decoder=Decoder(args),
        ))

        # 语言模型头：将 Decoder 的输出投影到词表空间
        # 输出维度为 vocab_size，用于预测下一个 token
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有模型参数
        self.apply(self._init_weights)

        # 打印模型参数数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=False):
        """
        统计模型参数数量aa

        Args:
            non_embedding: 如果为 True，不统计 embedding 层的参数

        Returns:
            参数总数
        """
        # 计算所有参数的数量
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding，就减去 embedding 的参数量
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        初始化模型权重

        使用正态分布初始化线性层和 Embedding 层的权重

        Args:
            module: 要初始化的模块
        """
        if isinstance(module, nn.Linear):
            # 线性层权重初始化为正态分布 N(0, 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 偏置初始化为 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding 层权重也初始化为正态分布 N(0, 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Transformer 前向传播

        Args:
            idx: 输入 token ids，形状 [batch_size, seq_len]
            targets: 目标 token ids（可选），用于训练时计算损失
                    形状 [batch_size, seq_len]

        Returns:
            logits: 模型输出的 logits，形状 [batch_size, seq_len, vocab_size] 或
                   [batch_size, 1, vocab_size]（推理时）
            loss: 交叉熵损失（训练时），或 None（推理时）
        """
        # 获取输入的设备和形状信息
        device = idx.device
        b, t = idx.size()  # b: batch_size, t: seq_len

        # 验证序列长度不超过最大限制
        assert t <= self.args.block_size, \
            f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # ===== 步骤 1: Token Embedding =====
        # 将 token ids 转换为词向量
        # 形状: [batch_size, seq_len] -> [batch_size, seq_len, n_embd]
        print("idx", idx.size())
        tok_emb = self.transformer.wte(idx)
        print("tok_emb", tok_emb.size())

        # ===== 步骤 2: 位置编码 =====
        # 添加位置信息
        # 形状保持: [batch_size, seq_len, n_embd]
        pos_emb = self.transformer.wpe(tok_emb)

        # ===== 步骤 3: Dropout =====
        x = self.transformer.drop(pos_emb)
        print("x after wpe:", x.size())

        # ===== 步骤 4: Encoder =====
        # 通过 Encoder 编码输入序列
        # 形状保持: [batch_size, seq_len, n_embd]
        enc_out = self.transformer.encoder(x)
        print("enc_out:", enc_out.size())

        # ===== 步骤 5: Decoder =====
        # Decoder 接收编码后的表示和 Encoder 的输出
        # 在这个简化实现中，Decoder 的输入也是 x（实际应用中可能是目标序列）
        # 形状保持: [batch_size, seq_len, n_embd]
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:", x.size())

        # ===== 步骤 6: 输出层 =====
        if targets is not None:
            # 训练模式：计算所有位置的 logits 和损失
            # 将 Decoder 输出投影到词表空间
            # 形状: [batch_size, seq_len, n_embd] -> [batch_size, seq_len, vocab_size]
            logits = self.lm_head(x)

            # 计算交叉熵损失
            # 将 logits 和 targets 展平为 2D 和 1D
            # logits: [batch_size * seq_len, vocab_size]
            # targets: [batch_size * seq_len]
            # ignore_index=-1: 忽略值为 -1 的位置（padding）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # 推理模式：只需要最后一个位置的 logits
            # 使用 [-1] 而不是 -1 来保持时间维度
            # 形状: [batch_size, 1, vocab_size]
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


def main():
    """
    主函数：演示如何使用 Transformer 模型

    这个示例展示了：
    1. 如何创建模型配置
    2. 如何使用 tokenizer 处理文本
    3. 如何运行模型前向传播
    4. 如何从输出中提取预测结果
    """
    # 创建模型配置
    # ModelArgs(n_embd, n_heads, dim, dropout, max_seq_len, vocab_size, block_size, n_layer)
    args = ModelArgs(
        n_embd=100,  # 嵌入维度
        n_heads=10,  # 注意力头数
        dim=100,  # 模型维度
        dropout=0.1,  # Dropout 比例
        max_seq_len=512,  # 最大序列长度
        vocab_size=1000,  # 词表大小（将由 tokenizer 更新）
        block_size=1000,  # 块大小
        n_layer=2  # Encoder/Decoder 层数
    )

    # 准备输入文本
    text = "我喜欢快乐地学习大模型"

    # 使用 BERT 中文 tokenizer 进行分词
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 将文本转换为 token ids
    inputs_token = tokenizer(
        text,
        return_tensors='pt',  # 返回 PyTorch 张量
        max_length=args.max_seq_len,  # 最大长度
        truncation=True,  # 超过最大长度时截断
        padding='max_length'  # 填充到最大长度
    )

    # 更新模型配置中的词表大小
    args.vocab_size = tokenizer.vocab_size

    # 创建 Transformer 模型
    transformer = Transformer(args)

    # 获取输入的 token ids
    inputs_id = inputs_token['input_ids']

    # 前向传播（推理模式，不计算损失）
    logits, loss = transformer.forward(inputs_id)

    # 打印输出 logits
    print("模型输出 logits:")
    print(logits)

    # 从 logits 中提取预测的 token id
    # argmax 找出概率最大的 token
    predicted_ids = torch.argmax(logits, dim=-1).item()

    # 将预测的 token id 解码回文本
    output = tokenizer.decode(predicted_ids)
    print("\n预测的输出:")
    print(output)


if __name__ == "__main__":
    print("=" * 50)
    print("开始运行 Transformer 示例")
    print("=" * 50)
    main()
    print("=" * 50)
    print("运行完成")
    print("=" * 50)
