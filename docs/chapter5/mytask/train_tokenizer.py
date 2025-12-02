"""
Tokenizer训练模块

本模块用于训练自定义的BPE (Byte Pair Encoding) tokenizer，用于语言模型训练。

主要功能：
1. 从JSONL格式的数据文件中读取训练文本
2. 使用BPE算法训练tokenizer
3. 配置特殊token（如BOS、EOS、UNK等）
4. 生成完整的tokenizer配置文件
5. 评估tokenizer的功能和性能

训练出的tokenizer将保存为HuggingFace兼容的格式，可以直接用于模型训练。
"""

import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

# 设置随机种子，确保结果可复现
random.seed(42)

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """
    从JSONL文件中读取文本数据

    这是一个生成器函数，逐行读取JSONL文件，提取每行的'text'字段。
    如果某行格式错误或缺少'text'字段，会跳过该行并打印错误信息。

    Args:
        file_path (str): JSONL格式的数据文件路径

    Yields:
        str: 从JSONL文件中提取的文本内容

    Example:
        JSONL文件格式：
        {"text": "这是第一段文本"}
        {"text": "这是第二段文本"}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析JSON格式的数据行
                data = json.loads(line)

                # 检查是否包含'text'字段
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")

                # 返回文本内容
                yield data['text']
            except json.JSONDecodeError:
                # JSON解析错误，跳过该行
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                # 缺少必要字段，跳过该行
                print(e)
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """
    创建完整的tokenizer配置文件

    生成HuggingFace兼容的tokenizer配置文件，包括：
    1. tokenizer_config.json: 主配置文件，包含所有tokenizer设置
    2. special_tokens_map.json: 特殊token映射文件

    Args:
        save_dir (str): 配置文件保存目录
    """
    # 主配置文件，包含tokenizer的所有设置
    config = {
        "add_bos_token": False,  # 不自动添加BOS token
        "add_eos_token": False,  # 不自动添加EOS token
        "add_prefix_space": False,  # 不在文本前添加空格
        "bos_token": "<|im_start|>",  # 句子开始token
        "eos_token": "<|im_end|>",  # 句子结束token
        "pad_token": "<|im_end|>",  # 填充token（使用EOS token）
        "unk_token": "<unk>",  # 未知token
        "model_max_length": 1000000000000000019884624838656,  # 最大序列长度（实际无限制）
        "clean_up_tokenization_spaces": False,  # 不清理tokenization后的空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # tokenizer类型
        # 聊天模板：用于将对话格式转换为模型输入格式
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建特殊token映射文件
    # 这个文件定义了所有特殊token及其对应的字符串
    special_tokens_map = {
        "bos_token": "<|im_start|>",  # 句子开始
        "eos_token": "<|im_end|>",  # 句子结束
        "unk_token": "<unk>",  # 未知token
        "pad_token": "<|im_end|>",  # 填充token
        "additional_special_tokens": ["<s>", "</s>"]  # 额外的特殊token
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """
    训练并保存自定义BPE tokenizer

    使用BPE (Byte Pair Encoding) 算法训练tokenizer，训练过程包括：
    1. 初始化BPE模型和预处理器
    2. 配置特殊token
    3. 从训练数据中学习词汇表
    4. 验证特殊token的ID映射
    5. 保存tokenizer和配置文件

    Args:
        data_path (str): 训练数据文件路径（JSONL格式）
        save_dir (str): tokenizer保存目录
        vocab_size (int): 词汇表大小，默认8192

    Raises:
        AssertionError: 如果特殊token的ID映射不符合预期
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # ==================== 初始化tokenizer ====================
    # 使用BPE模型，设置未知token为"<unk>"
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 设置文本规范化器：使用NFKC规范化（兼容性分解后规范化）
    # NFKC可以处理一些Unicode字符的变体形式
    tokenizer.normalizer = NFKC()

    # 设置预处理器：使用ByteLevel预处理器
    # ByteLevel将文本按字节级别处理，可以处理任何Unicode字符
    # add_prefix_space=False: 不在文本前添加空格
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 设置解码器：使用ByteLevel解码器，与预处理器对应
    tokenizer.decoder = decoders.ByteLevel()

    # ==================== 配置特殊token ====================
    # 定义所有特殊token，这些token会被优先添加到词汇表中
    # 顺序很重要：第一个token的ID为0，第二个为1，以此类推
    special_tokens = [
        "<unk>",        # 未知token (ID: 0)
        "<s>",          # 句子开始 (ID: 1)
        "</s>",         # 句子结束 (ID: 2)
        "<|im_start|>", # 对话开始标记 (ID: 3)
        "<|im_end|>"    # 对话结束标记 (ID: 4)
    ]

    # ==================== 配置训练器 ====================
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # 目标词汇表大小
        special_tokens=special_tokens,  # 特殊token列表
        min_frequency=2,  # 最小词频：出现次数少于2次的子词不会被加入词汇表
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始字母表（所有字节）
    )

    # ==================== 训练tokenizer ====================
    print(f"Training tokenizer with data from {data_path}")

    # 从JSONL文件中读取文本数据（生成器）
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    # length参数用于显示进度条，使用文件大小作为估计值
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # ==================== 验证特殊token映射 ====================
    # 确保特殊token的ID符合预期
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # ==================== 保存tokenizer ====================
    # 保存tokenizer的核心文件（包含词汇表和合并规则）
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # 创建配置文件（tokenizer_config.json和special_tokens_map.json）
    create_tokenizer_config(save_dir)

    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """
    评估tokenizer的功能和性能

    对训练好的tokenizer进行全面的功能测试，包括：
    1. 基本属性检查（词汇表大小、特殊token等）
    2. 聊天模板功能测试
    3. 编码解码一致性测试
    4. 特殊token处理测试

    Args:
        tokenizer_path (str): tokenizer保存路径
    """
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # ==================== 测试基本属性 ====================
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")  # 词汇表大小
    print(f"Special tokens: {tokenizer.all_special_tokens}")  # 所有特殊token
    print(f"Special token IDs: {tokenizer.all_special_ids}")  # 所有特殊token的ID

    # ==================== 测试聊天模板 ====================
    # 构造一个多轮对话示例
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]

    print("\n=== 聊天模板测试 ===")
    # 使用chat_template将对话格式转换为模型输入格式
    # tokenize=False: 返回文本而不是token ID
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True  # 如果启用，会在末尾添加assistant提示
    )
    print("Generated prompt:\n", prompt, sep="")

    # ==================== 测试编码解码 ====================
    print("\n=== 编码解码测试 ===")
    # 编码文本为token ID
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    # 解码token ID回文本（保留特殊token）
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    # 检查编码解码是否一致
    print("Decoded text matches original:", decoded == prompt)

    # ==================== 测试特殊token处理 ====================
    print("\n=== 特殊token处理 ===")
    # 测试包含特殊token的文本
    test_text = "<|im_start|>user\nHello<|im_end|>"
    # 编码
    encoded = tokenizer(test_text).input_ids
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    # 检查特殊token是否被正确保留
    print("Special tokens preserved:", decoded == test_text)

def main():
    """
    主函数：训练和评估tokenizer

    使用示例：
    1. 设置训练数据路径和保存目录
    2. 训练tokenizer
    3. 评估tokenizer功能
    """
    # ==================== 配置路径 ====================
    data_path = "your data path"  # 训练数据文件路径（JSONL格式）
    save_dir = "tokenizer_k"  # tokenizer保存目录

    # ==================== 训练tokenizer ====================
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144  # 词汇表大小：6144个token
    )

    # ==================== 评估tokenizer ====================
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()