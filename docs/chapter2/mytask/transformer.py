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
    dim: int  # 隐藏层维度
    vocab_size: int  # 词表大小
    n_head: int  # 多头注意力的头数
    n_layer: int  # Encoder/decoder 层数
    max_seq_len: int  # 最大序列长度,用于创建位置编码和掩码
    block_size: int  # 输出序列最大长度，通常等于max_sel_len
    dropout: float  # 概率值，防止过拟合


class PositionalEncoding(nn.Module):

    def __init__(self, args):
        super().__init__()
        # 创建位置编码矩阵，形状[block_size,n_emd]
        pe = torch.zeros(args.block_size, args.n_embd)
        # 创建位置索引[0,1,2,...,block-size-1],形状[block_size,1]
        # 创建一个从 0 到 args.block_size-1 的整数序列
        position = torch.arange(0, args.block_size).unsqueeze(1)

        # 计算位置编码的除数项（divisor term），用于实现 Transformer 论文中的位置编码公式
        # 公式: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # 其中 pos 是位置，i 是维度索引，d_model 是嵌入维度

        # 数学变换：1 / 10000^(2i/d_model) = exp(-2i/d_model * ln(10000))
        # 因此 div_term = exp(arange(0, d_model, 2) * -(ln(10000) / d_model))

        div_term = torch.exp(
            # torch.arange(0, args.n_embd, 2): 生成偶数索引 [0, 2, 4, 6, ...]，对应公式中的 2i
            # 形状: [n_embd/2]，例如 n_embd=8 时生成 tensor([0, 2, 4, 6])

            # -(math.log(10000.0) / args.n_embd): 缩放因子，值为 -ln(10000)/d_model
            # 负号使得频率随维度递减：低维度→高频率（捕捉短距离），高维度→低频率（捕捉长距离）

            # 整体相乘：arange * scale 得到指数值，例如 [0, -2.3, -4.6, -6.9]
            # torch.exp(): 对指数求幂，得到除数项，例如 [1.0, 0.1, 0.01, 0.001]
            # 这些除数让不同维度以不同频率编码位置信息
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 对偶数位置使用sin,对奇数位置是cos
        # pe[:, 0::2] 表示所有行，从第 0 列开始每隔 2 列取一个（即偶数列：0, 2, 4, 6, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] 表示所有行，从第 1 列开始每隔 2 列取一个（即奇数列：1, 3, 5, 7, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度：[block_size, n_embd] -> [1, block_size, n_embd]
        # 添加 batch 维度：在第 0 维插入大小为 1 的维度，使位置编码能够与 3D 输入数据 [batch_size, seq_len, n_embd] 相加
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
    归一化层（layer Normalization）
    与 Bach Layer 不同的是，layer Normalization 会对每个样本的所有维度的特征进行归一化
    而不是对整个 Bach 的某一个特征进行归一化，这使得更适合序列模型（序列可变）和小Bach
    公式: LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
    其中 μ 和 σ 是在特征维度上计算的均值和标准差
    args:
        features: 特征维度大小
        eps: 防止除0的小常数，默认1e-6
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 可学习的缩放参数γ(gamma),初始值为1
        self.a_2 = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数β(bete),初始值为0
        self.b_2 = nn.Parameter(torch.zeros(features))
        # epsilon: 防止除0的小数
        self.eps = eps

    def forward(self, x):
        # 计算最后一个维度(特征维度)的均值
        # keepdim: 保持维度，便于广播
        # mean: [bach_size,seq_len,1]
        mean = x.mean(-1, keepdim=True)

        # 计算最后一个维度的标准差
        # std: [bach_size,seq_len,1]
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
        assert args.dim % args.n_head == 0
        # 每个头的维度，等于模型隐藏层维度/头的总数
        # 例如：dim=512, n_heads=8, 则 head_dim=64
        self.head_dim = args.dim // args.n_head
        self.heads = args.n_head

        # 定义 Query,key,value 的线性变换矩阵
        # 输入维度：n_embd,输出维度 head_dim * n_heads
        # 这里通过一个大的线性层来代替 n_heads 个小的线性层

        # 1. 作用：将输入投影成 Q、K、V
        # 2. 输入维度：n_embd (如 512)
        # 3. 输出维度：n_heads * head_dim (如 8×64=512)
        # 4. bias=False：不使用偏置项
        # 5. 高效实现：一个大线性层代替 8 个小线性层
        # 6. 数学原理：矩阵乘法的结合律 (AB)C = A(BC)
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)

        # 输出投影矩阵，将多头的结果投影回原始维度
        # 维度：（n_heads * head_dim）-> dim
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        # 注意力权重的Dropout,用于防止过拟合

        # 多头注意力的输出（示例：512维向量）
        # output = [0.5, 0.3, -0.2, 0.8, ..., 0.1]  # 512个数
        #         ↑    ↑     ↑    ↑         ↑
        #       特征1 特征2 特征3 特征4 ... 特征512
        # 应用 Dropout (p=0.1)
        # 随机丢弃 10% 的特征维度
        # output = [0.56, 0, -0.22, 0.89, ..., 0]
        #         ↑     ↓    ↑     ↑         ↓
        #       保留  丢弃  保留  保留     丢弃
        #      (放大)            (放大)
        # 好处：
        # ✅ 防止过度依赖某些特征维度
        # ✅ 增强特征的鲁棒性
        # ✅ 配合残差连接，稳定训练

        self.attn_dropout = nn.Dropout(args.dropout)  # 📌 定义 Dropout 1
        # 残差连接前的Dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 📌 定义 Dropout 2
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
        # 只关心前两个维度，n_embd 是固定维度
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
        # xq.view() 是 PyTorch 张量的reshape（重塑形状）方法，用于改变张量的形状而不改变数据内容
        xq = xq.view(bsz, seqlen, self.heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.heads, self.head_dim)

        # 交换维度：将 n_heads 维度移到第二个位置
        # 从 (B, T, nh, hs) 变为 (B, nh, T, hs)
        # transpose（转置）操作，用于交换张量的两个维度,为了让每个注意力头可以独立并行计算,因为这个维度都一样
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


class EncoderLayer(nn.Module):
    """
    Encoder层
    每个Encoder层包含两个子层
    1. 多头自注意力机制
    2. 前馈神经网络
    每个子层都使用参差链接和归一化层
    结构：layerNorm->MultiHeadAttention->残差连接->layerNorm->FFM->残差连接
    """

    def __init__(self, args):
        super().__init__()
        # 第一个 layerNorm，在多头自注意力前
        self.attention_norm = LayerNorm(args.n_embd)

        # 为什么？因为输入句子是完整给出的！
        # 我们有全部信息，没有理由不让模型看到所有内容
        # 多头自注意力(不使用因果掩码，因为Encoder 可以看到所有位置)，因果掩码（Causal Mask / Attention Mask）是一种机制，用于防止模型在预测当前位置时"偷看"未来的信息。
        # # 任务：英文翻译成中文
        # 英文（输入） = "I love learning"
        # 中文（输出） = "我 爱 学习"
        #
        # # ==================== Encoder 处理输入 ====================
        # # Encoder 的任务：理解输入句子 "I love learning"
        #
        # input_tokens = ["I", "love", "learning"]
        #
        # # 当 Encoder 处理 "love" 这个词时：
        # # ✅ 可以看到 "I"（前面的词）
        # # ✅ 可以看到 "love"（当前的词）
        # # ✅ 可以看到 "learning"（后面的词）
        # MultiHeadAttention()

    pass


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # EncoderLayer


class Decoder(nn.Module):
    pass


class Transformer(nn.Module):
    #     ↑          ↑
    #   类名      父类（继承自）
    """
    完整的transformer 模型
    包含Encoder 和 Decoder的完整的transformer架构，适用于序列到序列任务
    模型结构
    输入->embedding-位置编码->encoder->decoder->输出投影->logits
        输出投影是一个线性变换层（nn.linear）,作用是将 Decoder 的输出向量映射到词表空间
        logits 是输出投影后得到的原始得分（未归一化的概率），表示每个词的"可能性得分"
    Args:
        args:模型参数
    """

    def __init__(self, args):
        super().__init__()
        assert args.vocab_size is not None, "必须指定词表大小"
        assert args.block_size is not None, "必须指定最大序列长度"
        self.args = args

        # 构建transformer的各个组件
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层，将token id转为向量
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            # 位置编码,添加位置信息
            wpe=PositionalEncoding(args),
            # Dropout层
            dorp=nn.Dropout(args.dropout),
            # Encoder 模块
            encoder=Encoder(args),
            # Decoder 模块
            decoder=Decoder(args)
        ))


def main():
    args = ModelArgs(
        n_embd=100,
        dim=100,
        vocab_size=1000,
        n_head=10,
        n_layer=2,
        max_seq_len=512,
        block_size=1000,
        dropout=0.1
    )

    text = "我喜欢快乐的学习大模型"
    # BertTokenizer 是 huggingface transformers 库里用于对文本进行分词和编码的分词器类
    # from_pretrained 是其常用的类方法，用于加载官方预训练好的分词器配置和词表
    # 例如这里加载"bert-base-chinese"模型的分词器，可以将中文句子切分成BERT需要的token id
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 这段代码的参数作用解释如下：
    # - text: 需要分词和编码的原始文本。
    # - return_tensors='pt': 返回结果以 PyTorch 的 tensor 形式；适用于后续模型输入。
    # - max_length: 指定输出的序列最大长度（超过会被截断，短于则补齐）；这里用 args.max_seq_len 控制。
    # - truncation=True: 超过 max_length 的文本会被截断，防止序列过长。
    # - padding='max_length': 不足 max_length 的输入会自动填充（pad）到统一长度，便于批量处理。
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_seq_len,
        truncation=True,
        padding='max_length'
    )

    # 因为有时我们初始化ModelArgs时还没加载分词器（tokenizer），这时暂时给vocab_size赋一个默认值。
    # 加载分词器后，可以获得其实际词表大小（tokenizer.vocab_size），然后再更新到args里，确保模型和分词器词表数量一致。
    args.vocab_size = tokenizer.vocab_size

    transformer = Transformer(args)


if __name__ == "main__":
    main()
