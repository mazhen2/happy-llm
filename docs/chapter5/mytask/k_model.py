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
