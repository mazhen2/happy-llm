import torch
from transformers import PreTrainedModel, AutoTokenizer
from transformers import PretrainedConfig
from typing import Any, Optional, Tuple
from torch import nn


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


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

    pass


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()

    pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

    pass


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
        out = x + self.feed_forward.forward(self.ffn_norm(h))

        return out


class Transformer(PreTrainedModel):
    """
    Tink_k Transformer参数
    完整的Transformer解码器架构，包含
    1. 词嵌入层（Token Embedding）
    2. 多层解码器（Decoder Layers）
    3. 输出层（output layer）
    4. 权重共享（Embedding 和 output 共享权重）
    5. 旋转位置编码（RoPE）

    模型流程：
        token->Embedding->Dropout->DecoderLayers->RMSNorm->output-Logits
    """

    config_class = ModelConfig  # 配置类，用于保存和加载模型配置
    last_loss: Optional[torch.tensor]  # 记录最后一次损失值，用于调试

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

    pass


if __name__ == '__main__':
    """
    模型测试和示例
    演示如何使用Tink-k模型进行前向传播
    1. 加载tokenizer
    2. 创建模型配置和实例
    3. 计算模型参数量
    4. 准备输入数据
    5. 进行前向传播
    """

    # 加载tokenizer,用于将文本转换为token ids
    AutoTokenizer.from_pretrained("tokenize_k")

    # 创建模型配置
    # dim = 1024:模型隐藏层维度1024
    # n_layer = 18:使用18层Transformer
    # 其他参数使用默认值
    args = ModelConfig(
        dim=1024,
        n_layers=18,
    )

    model = Transformer(args=args)
