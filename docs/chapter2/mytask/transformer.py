import math
from dataclasses import dataclass
import torch
from torch import nn
from transformers import BertTokenizer


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


class Encoder(nn.Module):
    pass


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
