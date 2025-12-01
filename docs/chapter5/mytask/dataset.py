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


class SFTDataset(Dataset):
    """
    监督微调(Supervised Fine-Tuning, SFT)数据集类

    用于语言模型的监督微调阶段，处理对话格式的数据。数据格式为JSONL文件，
    每行包含一个JSON对象，其中包含对话消息列表，格式为：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

    功能特点：
    - 使用chat_template将对话格式转换为模型输入格式
    - 只对assistant回复部分计算损失（通过loss_mask实现）
    - 自动截断或填充到指定最大长度
    - 生成输入序列(X)和目标序列(Y)，用于自回归训练

    Attributes:
        data_path (str): JSONL格式的训练数据文件路径
        tokenizer: 分词器对象，用于文本编码
        max_length (int): 序列的最大长度，默认512
        padding (int): 填充token的ID，默认0
        data (list): 加载的原始数据列表
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化SFT数据集

        Args:
            data_path (str): JSONL格式的训练数据文件路径
            tokenizer: 分词器对象，用于将文本转换为token ID
            max_length (int): 序列的最大长度，超过此长度会被截断，不足会填充
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0  # padding token的ID，通常为0
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

    def generate_loss_mask(self, input_ids):
        """
        生成损失掩码，用于标记哪些位置需要计算损失

        在SFT训练中，我们只对assistant的回复部分计算损失，而忽略：
        - system消息
        - user消息
        - padding位置

        实现逻辑详解：

        1. 初始化所有位置为0（不计算损失）
           所有位置默认不计算损失，后续只标记assistant回复部分

        2. 查找所有"<|im_start|>assistant\n"标记的位置

           "<|im_start|>assistant\n" 是什么？
           - 这是chat_template中用于标记assistant回复开始的特殊标记
           - 完整的对话格式示例：

             <|im_start|>system\n你是一个AI助手<|im_end|>\n
             <|im_start|>user\n你好<|im_end|>\n
             <|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>\n

           - "<|im_start|>assistant\n" 标记了assistant回复的开始
           - "<|im_end|>" (eos_token) 标记了assistant回复的结束

           编码后的token序列可能是：
           [3, 456, 789, ..., 123, 234, ..., 4, ..., 567, 890, ..., 4]
           |<--system-->|  |<--user-->|  |<--assistant-->|
           其中 3 可能是 <|im_start|>，4 是 <|im_end|>

        3. 对于每个assistant标记，将其后的内容（直到遇到eos_token）标记为1

           这句话的含义：
           - 找到 "<|im_start|>assistant\n" 在token序列中的位置（假设位置为 i）
           - 从 assistant 标记结束的位置开始（i + assistant标记长度）
           - 向后查找，直到找到 eos_token（<|im_end|>，假设位置为 j）
           - 将位置 [i + assistant标记长度, j] 范围内的所有token标记为1
           - 这表示：只有assistant回复的内容需要计算损失

           示例：
           假设完整的token序列是：
           [3, 456, 789, 4, 3, 123, 234, 4, 3, 567, 890, 4, 0, 0]
           |<--system-->|  |<--user-->|  |<--assistant-->|  |padding|

           查找过程：
           - 找到第一个 "<|im_start|>assistant\n" 的位置（假设是位置8）
           - 从位置 8 + assistant标记长度 开始查找
           - 找到对应的 eos_token（<|im_end|>）位置（假设是位置11）
           - 将位置 [9, 10, 11] 标记为1（assistant回复的内容）

           最终 loss_mask：
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
           |<--system-->|  |<--user-->|  |<--assistant-->|  |padding|
           不计算损失    不计算损失      计算损失          不计算损失

        4. 这样只有assistant的回复部分会参与损失计算
           训练目标：让模型学会生成正确的assistant回复
           不需要学习：system和user的消息（这些是输入，不是要生成的内容）

        Args:
            input_ids (list): token ID序列

        Returns:
            list: 损失掩码列表，1表示计算损失，0表示忽略
        """
        # 初始化所有位置为0（默认不计算损失）
        mask = [0] * len(input_ids)

        # 获取assistant标记的token序列
        # 格式：<|im_start|>assistant\n
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0

        # 遍历整个序列，查找所有assistant标记
        # i 是当前检查的起始位置，从0开始，逐步向后移动
        while i <= n - a_length:
            # 检查当前位置是否匹配assistant标记序列
            match = True
            for k in range(a_length):
                # 逐个比较token ID
                # 如果发现不匹配，立即停止比较
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break

            if match:
                # 找到assistant标记后，查找对应的结束标记(eos_token_id)
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx  # 找到eos_token的位置
                        break

                if j is not None:
                    # 标记assistant回复的内容（从assistant标记后到eos_token，包含eos_token）
                    start = i + a_length  # assistant标记后的第一个位置
                    end = j  # eos_token的位置

                    # 将assistant回复部分标记为1（需要计算损失）
                    # 这包括assistant回复的所有内容，以及eos_token本身
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1

                # 跳过当前assistant标记，避免重叠匹配
                i += a_length
            else:
                # 不匹配，继续向后查找
                i += 1

        return mask


def __getitem__(self, index: int):
    """
    获取指定索引的数据样本

    处理流程：
    1. 解析JSON数据，提取对话消息
    2. 使用chat_template将对话格式转换为模型输入格式
    3. 使用tokenizer编码文本为token ID序列
    4. 截断或填充到max_length
    5. 生成损失掩码，只对assistant回复部分计算损失
    6. 生成输入序列X和目标序列Y

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

    # 使用chat_template将对话格式转换为模型输入格式
    # tokenize=False: 返回文本而不是token ID
    # add_generation_prompt=False: 不添加生成提示
    text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)

    # 使用tokenizer编码文本，并截断到最大长度
    input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
    text_len = len(input_id)

    # 计算需要填充的长度（如果文本长度小于max_length）
    padding_len = self.max_length - text_len

    # 在序列末尾添加padding token
    input_id = input_id + [self.padding] * padding_len

    # 生成损失掩码：只对assistant回复部分计算损失
    loss_mask = self.generate_loss_mask(input_id)

    # 转换为numpy数组
    input_id = np.array(input_id)

    # 生成输入序列X（前n-1个token）和目标序列Y（后n-1个token）
    X = np.array(input_id[:-1]).astype(np.int64)
    Y = np.array(input_id[1:]).astype(np.int64)

    # 损失掩码也需要相应调整（去掉第一个位置）
    loss_mask = np.array(loss_mask[1:]).astype(np.int64)

    # 转换为PyTorch张量并返回
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
