"""
Transformer 完整实现

本文件实现了完整的 Transformer 模型，包括：
- 编码器（Encoder）：理解输入序列
- 解码器（Decoder）：生成输出序列
- 多头注意力机制（Multi-Head Attention）
- 位置编码（Positional Encoding）
- 层归一化（Layer Normalization）

基于论文 "Attention is All You Need" (Vaswani et al., 2017)
适用于序列到序列（Seq2Seq）任务，如机器翻译、文本摘要等
"""

import math
from dataclasses import dataclass
import torch
from torch import nn
from transformers import BertTokenizer
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    模型配置参数类

    参数类别与其控制的模型方面说明：

    📐【维度参数】—— 控制模型“宽度”
        - n_embd: 词嵌入向量维度，决定每个token被表示成多长的向量；向量维度越大，每个单词的表征能力越强。
        - dim: Transformer各层隐藏状态的维度，常等于n_embd。它代表了每一层信息流通过的通道数，通道数越大，网络每一层能同时保留、表达的信息也越丰富。模型“宽度”指的就是网络每层这一维度的大小，例如dim=1024表示每一层有1024维响应。宽度增大时，模型可以表达更复杂的特征，但参数量和计算消耗也提升。
        - vocab_size: 词表大小，决定模型能区分和处理多少个不同token。

    🔢【结构参数】—— 控制模型“深度”和“头数”
        - n_heads: 多头注意力头数。每个注意力头可以在不同子空间独立学习信息；头数越多，模型捕捉多样关系的能力越强，但需要dim能被n_heads整除。
        - n_layer: 层数，也叫深度，通常指Encoder或Decoder堆叠的子层数。层越多，模型信息加工能力越强，能实现更高层次的抽象，但计算和显存消耗也更大。

    📏【长度参数】—— 控制序列长度
        - max_seq_len: 最大输入序列长度，限制模型能处理的最长序列（用于位置编码和注意力掩码）。
        - block_size: 实际输入切分的最大长度，通常与max_seq_len相同。

    🛡️【正则化参数】—— 防止过拟合
        - dropout: Dropout概率，训练时随机丢弃一部分神经元的输出，有效缓解过拟合，提升泛化能力。

    参数影响模型能力的本质解释：
      - “宽度”参数（如dim、n_embd）决定了每层神经元（特征）的数量，可以并行表示和处理的信息量；就像一条高速公路车道数越多，能同时通过的车辆就越多。
      - “深度”参数（如n_layer）对应模型有多少层，每一层都对信息进一步处理和提炼，层数多则模型具备更复杂的非线性表达能力。
      - “头数”参数让注意力机制在多个子空间并行工作，帮助模型捕捉不同类型的特征关系。
      - “长度”参数（如max_seq_len）简单规定了能处理的序列最大范围，超限部分会被截断。
      - Dropout等正则参数是模型为防止学习到训练中的偶然规律（过拟合）而设置的抑制装置。

    常见配置示例：
        - 小模型：n_embd=256, dim=256, n_heads=4, n_layer=4, max_seq_len=512
        - BERT-base：n_embd=768, dim=768, n_heads=12, n_layer=12, max_seq_len=512
        - GPT-2-small：n_embd=768, dim=768, n_heads=12, n_layer=12, max_seq_len=1024
    """
    n_embd: int  # 词向量维度
    n_heads: int  # 多头注意力的头数
    dim: int  # 隐藏层维度
    dropout: float  # 概率值，防止过拟合
    max_seq_len: int  # 最大序列长度,用于创建位置编码和掩码
    vocab_size: int  # 词表大小
    block_size: int  # 输出序列最大长度，通常等于max_sel_len
    n_layer: int  # Encoder/decoder 层数


class PositionalEncoding(nn.Module):
    """
    位置编码（Positional Encoding）
    
    由于 Transformer 没有 RNN 的顺序处理特性，需要通过位置编码来注入位置信息。
    使用正弦和余弦函数生成位置编码，具有以下优点：
    1. 可以处理任意长度的序列
    2. 不同位置的相对位置关系可以被模型学习
    3. 不需要训练，是固定的
    
    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中 pos 是位置，i 是维度索引，d_model 是嵌入维度
    
    Args:
        args: 模型配置参数
    """

    def __init__(self, args):
        super().__init__()
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
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    与 Batch Normalization 不同，Layer Norm 对每个样本的所有特征进行归一化，
    而不是对整个 batch 的同一个特征进行归一化。这使得它更适合序列模型和小 batch。
    
    公式：LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
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
        # 校验：隐藏层是多头自注意力头数的整数倍，因为我们后面会将输入平均拆分为 n_heads 个子空间
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型隐藏层维度/头的总数
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


class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network / MLP）
    
    在 Transformer 中，每个 Encoder/Decoder 层都包含一个前馈网络。
    它由两个线性层组成，中间使用 ReLU 激活函数。
    
    结构：Linear -> ReLU -> Linear -> Dropout
    
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
        # 数据流：x -> Linear(w1) -> ReLU -> Linear(w2) -> Dropout -> output
        # 这个两层结构允许模型学习更复杂的非线性变换
        return self.dropout(self.w2(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """
    Encoder 层
    
    每个 Encoder 层包含两个子层：
    1. 多头自注意力机制（Multi-Head Self-Attention）
    2. 前馈神经网络（Feed-Forward Network）
    
    每个子层都使用残差连接（Residual Connection）和层归一化（Layer Normalization）
    结构：LayerNorm -> MultiHeadAttention -> 残差连接 -> LayerNorm -> FFN -> 残差连接
    
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
        # ========== 子层 1: 掩码自注意力 + 残差连接 ==========
        # 目的：让 Decoder 理解"已经生成的内容"，但不能看到"未来的内容"
        # 例如：翻译时生成 "我爱学习"，当前生成到 "学习" 时，只能看到 "我" 和 "爱"

        # 步骤1: LayerNorm 归一化（Pre-LN 架构：先归一化再计算）
        x_norm = self.attention_norm_1(x)
        # x_norm 形状: [batch_size, tgt_seq_len, n_embd]

        # 步骤2: 掩码自注意力（is_causal=True 使用因果掩码）
        # Q、K、V 都来自 Decoder 输入（自注意力）
        # 因果掩码确保：位置 i 只能看到位置 0, 1, ..., i（不能看到 i+1 及之后）
        # 这样训练时模拟真实生成过程（逐词生成，不能偷看答案）
        masked_attn_output = self.mask_attention.forward(x_norm, x_norm, x_norm)

        # 步骤3: 残差连接（Residual Connection）
        # 将原始输入 x 加上注意力的输出，保留原始信息并增加新特征
        # 残差连接的好处：缓解梯度消失，让信息流动更顺畅
        x = x + masked_attn_output
        # x 形状: [batch_size, tgt_seq_len, n_embd]

        # ========== 子层 2: 交叉注意力 + 残差连接 ==========
        # 目的：让 Decoder 关注"输入序列"，实现源语言和目标语言的对齐
        # 例如：生成中文 "学习" 时，应该主要关注英文的 "learning"

        # 步骤1: LayerNorm 归一化
        x_norm = self.attention_norm_2(x)
        # x_norm: 第1个子层的输出（已包含自注意力信息）

        # 步骤2: 交叉注意力（Cross-Attention）
        # Query: 来自 Decoder（x_norm），表示"我想找什么"
        # Key, Value: 来自 Encoder 输出（enc_out），表示"输入序列有什么"
        # is_causal=False: 可以看到 Encoder 的所有位置（输入是完整给出的）
        #
        # 这就是 Decoder 如何"理解输入"并"生成对应输出"的关键机制！
        # 交叉注意力权重反映了输入输出的对齐关系（Alignment）
        cross_attn_output = self.attention.forward(x_norm, enc_out, enc_out)
        #                                          ↑       ↑       ↑
        #                                        Query    Key    Value
        #                                       Decoder  Encoder Encoder

        # 步骤3: 残差连接
        # 将第1个子层的输出 x 加上交叉注意力的输出
        h = x + cross_attn_output
        # h 形状: [batch_size, tgt_seq_len, n_embd]
        # h 现在包含：①已生成内容的信息（子层1） + ②输入序列的信息（子层2）

        # ========== 子层 3: 前馈网络 + 残差连接 ==========
        # 目的：对每个位置的表示进行独立的非线性变换，增强特征表达能力
        # 前馈网络结构：Linear(扩展) -> ReLU -> Linear(压缩) -> Dropout

        # 步骤1: LayerNorm 归一化
        h_norm = self.ffn_norm(h)

        # 步骤2: 前馈网络
        # 每个词的表示独立处理（没有位置间的交互）
        # 通过两层线性变换和激活函数，增加模型的非线性表达能力
        ffn_output = self.feed_forward.forward(h_norm)

        # 步骤3: 残差连接
        out = h + ffn_output
        # out 形状: [batch_size, tgt_seq_len, n_embd]
        # out 包含：①自注意力信息 + ②交叉注意力信息 + ③前馈网络的变换
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


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    包含 Encoder 和 Decoder 的完整 Transformer 架构，适用于序列到序列任务。
    模型结构：
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
            # 词嵌入层 (Word Token Embedding)：将 token ID 转换为向量
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            # 位置编码 (Word Position Embedding)：添加位置信息
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
        统计模型参数数量
        
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
    # ===== 步骤 1: 创建模型配置 =====
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

    # ===== 步骤 2: 准备输入文本 =====
    text = "我喜欢快乐的学习大模型"

    # ===== 步骤 3: 使用 tokenizer 处理文本 =====
    # 使用 BERT 中文 tokenizer 进行分词
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 将文本转换为 token IDs
    # 参数说明：
    #   - text: 需要分词和编码的原始文本
    #   - return_tensors='pt': 返回 PyTorch 张量格式
    #   - max_length: 序列最大长度（超过会被截断）
    #   - truncation=True: 启用截断
    #   - padding='max_length': 填充到最大长度
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_seq_len,
        truncation=True,
        padding='max_length'
    )

    # ===== 步骤 4: 更新模型配置 =====
    # 更新 vocab_size 为 tokenizer 的实际词表大小
    # 确保模型和分词器的词表数量一致
    args.vocab_size = tokenizer.vocab_size

    # ===== 步骤 5: 创建 Transformer 模型 =====
    transformer = Transformer(args)

    # ===== 步骤 6: 前向传播 =====
    # 获取输入的 token IDs
    inputs_id = inputs_token['input_ids']

    # 前向传播（推理模式，不计算损失）
    logits, loss = transformer.forward(inputs_id)

    # ===== 步骤 7: 处理输出 =====
    print("\n" + "="*50)
    print("模型输出结果")
    print("="*50)
    
    # 打印 logits 的形状信息
    print(f"Logits 形状: {logits.shape}")
    print(f"  解释: [batch_size={logits.size(0)}, seq_len={logits.size(1)}, vocab_size={logits.size(2)}]")
    
    # 从 logits 中提取预测的 token ID
    # argmax 找出概率最高的 token
    predicted_id = torch.argmax(logits, dim=-1).squeeze().item()
    print(f"\n预测的 token ID: {predicted_id}")
    
    # 解码单个 token
    predicted_token = tokenizer.decode([predicted_id])
    print(f"预测的 token: {predicted_token}")
    
    # 注意事项
    print("\n" + "="*50)
    print("⚠️  注意")
    print("="*50)
    print("1. 模型是随机初始化的，未经过训练")
    print("2. 预测结果是随机的，没有实际意义")
    print("3. '##' 前缀表示这是一个子词（subword）")
    print("4. 需要训练模型才能得到有意义的预测")
    
    # 额外展示：输入文本的 token 化结果
    print("\n" + "="*50)
    print("输入文本的 Tokenization 结果")
    print("="*50)
    print(f"原始文本: {text}")
    print(f"Token IDs: {inputs_id[0][:20].tolist()}...")  # 只显示前20个
    
    # 解码查看前几个 token
    print(f"\n前5个 token 的解码:")
    for i in range(min(5, inputs_id.size(1))):
        token_id = inputs_id[0, i].item()
        token_text = tokenizer.decode([token_id])
        print(f"  位置 {i}: ID={token_id:5d} → '{token_text}'")


if __name__ == "__main__":
    print("=" * 50)
    print("开始运行 Transformer 示例")
    print("=" * 50)
    main()
    print("=" * 50)
    print("运行完成")
    print("=" * 50)
