"""
数据集处理模块

本模块用于处理原始数据集，将其转换为模型训练所需的格式。

主要功能：
1. 处理预训练数据：将长文本切分成固定长度的块
2. 处理SFT数据：将原始对话格式转换为标准对话格式

数据格式说明：
- 预训练数据：每行一个JSON对象，包含'text'字段
- SFT数据：每行一个JSON对象，包含对话消息列表
"""

import os
import json
from tqdm import tqdm

# ==================== 配置路径 ====================
# pretrain_data: 运行download_dataset.sh时下载的pretrain_data本地路径
pretrain_data = 'D:\datasets\chapter5'
# 处理后的预训练数据输出路径
output_pretrain_data = 'D:\datasets\chapter5'

# sft_data: 运行download_dataset.sh时下载的sft_data本地路径
sft_data = 'D:\datasets\chapter5\BelleGroup'
# 处理后的SFT数据输出路径
output_sft_data = 'D:\datasets\chapter5\BelleGroup'

# ==================== 1. 处理预训练数据 ====================
def split_text(text, chunk_size=512):
    """
    将文本按指定长度切分成块

    预训练数据通常包含很长的文本，需要切分成固定长度的块以便训练。
    每个块的长度为chunk_size，最后一个块可能小于chunk_size。

    Args:
        text (str): 待切分的文本
        chunk_size (int): 每个块的长度，默认512

    Returns:
        list: 文本块列表，每个元素是一个字符串

    Example:
        >>> split_text("Hello world", chunk_size=5)
        ['Hello', ' worl', 'd']
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 处理预训练数据：读取原始数据，切分文本，保存为JSONL格式
with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # 使用tqdm显示处理进度
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):
            # 解析JSON格式的数据行
            line = json.loads(line)
            # 提取文本内容
            text = line['text']

            # 将长文本切分成固定长度的块
            chunks = split_text(text)

            # 将每个块写入输出文件
            for chunk in chunks:
                # 每个块保存为一个JSON对象，包含'text'字段
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# ==================== 2. 处理SFT数据 ====================
def convert_message(data):
    """
    将原始SFT数据转换为标准对话格式

    原始数据格式：
    [
        {"from": "human", "value": "..."},
        {"from": "assistant", "value": "..."},
        ...
    ]

    转换后的标准格式：
    [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

    Args:
        data (list): 原始对话数据列表，每个元素包含'from'和'value'字段

    Returns:
        list: 标准格式的对话消息列表，每个元素包含'role'和'content'字段
    """
    # 初始化消息列表，添加system消息
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]

    # 遍历原始数据，转换为标准格式
    for item in data:
        if item['from'] == 'human':
            # 人类消息转换为user角色
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            # 助手消息转换为assistant角色
            message.append({'role': 'assistant', 'content': item['value']})

    return message

# 处理SFT数据：读取原始数据，转换格式，保存为JSONL格式
with open(output_sft_data, 'a', encoding='utf-8') as sft:
    with open(sft_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # 使用tqdm显示处理进度
        for item in tqdm(data, desc="Processing", unit="lines"):
            # 解析JSON格式的数据行
            item = json.loads(item)

            # 转换对话格式
            message = convert_message(item['conversations'])

            # 将转换后的消息写入输出文件
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')