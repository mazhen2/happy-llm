"""
模型导出模块

本模块用于将训练好的模型导出为HuggingFace格式，方便后续使用和部署。

主要功能：
1. 加载训练好的模型检查点
2. 注册自定义模型类到HuggingFace的自动类系统
3. 将模型和tokenizer保存为HuggingFace兼容的格式
4. 导出后的模型可以使用AutoModelForCausalLM.from_pretrained()加载

导出格式：
- model.safetensors 或 pytorch_model.bin: 模型权重文件
- config.json: 模型配置文件
- tokenizer相关文件: tokenizer.json, tokenizer_config.json等
"""

import torch
import warnings
from transformers import AutoTokenizer
from k_model import Transformer, ModelConfig

# 忽略用户警告
warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    """
    统计模型中可训练参数的数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_model(tokenizer_path, model_config, model_ckpt_path, save_directory):
    """
    导出模型为HuggingFace格式
    
    将训练好的模型检查点转换为HuggingFace兼容的格式，包括：
    1. 注册自定义模型类
    2. 加载模型权重
    3. 保存模型和tokenizer
    
    Args:
        tokenizer_path (str): tokenizer保存路径
        model_config (ModelConfig): 模型配置对象
        model_ckpt_path (str): 模型检查点文件路径（.pth文件）
        save_directory (str): 导出模型的保存目录
    """
    # ==================== 注册自定义类和配置 ====================
    # 将自定义的ModelConfig注册到HuggingFace的自动类系统
    # 这样可以使用AutoConfig.from_pretrained()加载配置
    ModelConfig.register_for_auto_class()
    
    # 将自定义的Transformer模型注册为AutoModelForCausalLM
    # 这样可以使用AutoModelForCausalLM.from_pretrained()加载模型
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    # ==================== 初始化模型 ====================
    # 根据配置创建模型实例
    model = Transformer(model_config)
    
    # 确定设备（优先使用CUDA）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================== 加载模型权重 ====================
    # 加载检查点文件
    state_dict = torch.load(model_ckpt_path, map_location=device)
    
    # 移除可能存在的多余前缀（来自某些训练框架，如torch.compile）
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # 加载权重到模型（strict=False允许部分权重不匹配）
    model.load_state_dict(state_dict, strict=False)
    
    # 打印模型参数量
    print(f'模型参数: {count_parameters(model)/1e6:.2f}M = {count_parameters(model)/1e9:.2f}B')

    # ==================== 加载tokenizer ====================
    # 加载tokenizer（与模型配套使用）
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,  # 信任远程代码（如果tokenizer有自定义代码）
        use_fast=False  # 不使用快速tokenizer
    )

    # ==================== 保存模型和tokenizer ====================
    # 保存模型为HuggingFace格式
    # safe_serialization=False: 使用pytorch_model.bin而不是safetensors格式
    model.save_pretrained(save_directory, safe_serialization=False)
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_directory)
    
    print(f'模型和tokenizer已保存至: {save_directory}')


if __name__ == '__main__':
    """
    主函数：导出模型示例
    
    使用示例：
    1. 配置模型参数（dim, n_layers等）
    2. 指定tokenizer路径、模型检查点路径和保存目录
    3. 调用export_model函数导出模型
    """
    # ==================== 模型配置 ====================
    # 定义模型配置，需要与训练时使用的配置一致
    config = ModelConfig(
        dim=1024,      # 模型维度
        n_layers=18,   # Transformer层数
    )

    # ==================== 导出模型 ====================
    export_model(
        tokenizer_path='./tokenizer_k/',  # tokenizer保存路径
        model_config=config,  # 模型配置
        model_ckpt_path='./BeelGroup_sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth',  # 模型检查点路径
        save_directory="k-model-215M"  # 导出模型的保存目录
    )