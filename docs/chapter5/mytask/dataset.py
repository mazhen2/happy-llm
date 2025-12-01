import json
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    """
    预训练数据集

    用于语言预训练阶段，处理纯文本数据，数据格式为json
    每行包含一个json对象，其中text字段为待训练的文本内容

    功能特点：
        自动添加BOS token到文本开头
        自动截断或填充到指定最大长度
        生成输入序列（x）和目标序列（Y）,用于自回归训练
        生成损失掩码，忽略padding位置的损失计算

    属性：
        data(list): 加载的原始数据列表
        data_path(str)：json格式的预训练数据文件路径
        tokenizer: 分词器对象，用于文本编码
        max_length(int): 序列的最大长度，默认512
        padding(int): 填充token的ID,默认为0
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集

        Args:
            data_path (str): JSONL格式的训练数据文件路径
            tokenizer: 分词器对象，用于将文本转换为token ID
            max_length (int): 序列的最大长度，超过此长度会被截断，不足会填充
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        # 读取所有数据行到内存
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        """
        返回数据集的大小

        Returns:
            int: 数据集中的样本数量
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        获取指定索引的数据样本

        处理流程:
        1. 解析json数据，提取文本内容
        2. 添加BOS token到文本开头
           - BOS (Beginning of Sentence) token 是"句子开始"标记
           - 作用：
             * 标记序列的开始位置，帮助模型识别文本的起始
             * 在生成任务中，BOS token可以作为生成起点
             * 统一序列格式，所有文本都以BOS token开始
           - 在这个项目中，BOS token通常是 "<|im_start|>" 或 "<s>"
           示例："这是一个示例文本" -> "<|im_start|>这是一个示例文本"
        3. 使用tokenizer编码文本为token ID序列
        4. 截断或填充到max_length
        5. 生成输入序列x(前n-1个token)和目标序列（后n-1个token）
        6. 生成损失掩码，标记哪些位置需要计算损失

        Args:
            index (int): 样本索引

        Returns:
            tuple: (X, Y, loss_mask)
                - X (torch.Tensor): 输入序列，shape为[max_length-1]
                - Y (torch.Tensor): 目标序列，shape为[max_length-1]
                - loss_mask (torch.Tensor): 损失掩码，1表示计算损失，0表示忽略
        """

        # 解析JSON格式的数据行
        sample = json.loads(self.data[index])

        # 在文本开头添加BOS (Beginning of Sentence) token
        # - sample['text']: 从解析的JSON对象中提取'text'字段的值（原始文本内容）
        # - f"{...}{...}": Python的f-string格式化，将两个字符串拼接在一起
        text = f"{self.tokenizer.bos_token}{sample['text']}"

        # 使用tokenizer编码文本，并截断到最大长度
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)

        # 计算需要填充的长度（如果文本长度小于max_length）
        padding_len = self.max_length - text_len

        # 在序列末尾添加padding token
        input_id = input_id + [self.padding] * padding_len

        # 生成损失掩码：1表示真实token（需要计算损失），0表示padding（忽略损失）
        loss_mask = [1] * text_len + [0] * padding_len

        # 转换为numpy数组
        input_id = np.array(input_id)

        # 生成输入序列X（前n-1个token）和目标序列Y（后n-1个token）
        # 这是自回归语言模型的标准做法：给定前n-1个token，预测后n-1个token
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)

        # 损失掩码也需要相应调整（去掉第一个位置，因为X和Y都少了一个元素）
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        # 转换为PyTorch张量并返回
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

