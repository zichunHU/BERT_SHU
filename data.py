# data.py
# 此模块负责加载 IMDB 数据集，对文本进行分词、填充、截断处理，并进行训练集与验证集的划分。
# 同时提供将处理后的数据集保存到磁盘（save_to_disk）或从磁盘加载（load_from_disk），
# 以保证每次使用完全相同的划分，方便控制变量和复现实验。

import os
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer

def load_and_preprocess(max_length=128, test_size=0.2, seed=42, save=False, load_saved=False):
    """
    加载并预处理 IMDB 数据集，同时支持保存或加载已处理数据集。

    参数:
        max_length: 每条文本的最大长度（超过截断，不足则填充）
        test_size: 训练集中用于验证的比例
        seed: 随机种子，确保划分结果可复现
        save: 是否将处理后的数据集保存到磁盘（True/False）
        load_saved: 是否加载已保存的数据集（True/False）
    返回:
        train_dataset, val_dataset, test_dataset, tokenizer
    """

    # 如果 load_saved 为 True 并且已保存数据集存在，则直接加载
    if load_saved and os.path.exists("train_dataset") and os.path.exists("val_dataset") and os.path.exists("test_dataset"):
        print("Loading datasets from disk...")
        train_dataset = load_from_disk("train_dataset")
        val_dataset = load_from_disk("val_dataset")
        test_dataset = load_from_disk("test_dataset")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return train_dataset, val_dataset, test_dataset, tokenizer

    # 否则，从 Hugging Face Hub 加载原始 IMDB 数据集
    print("Loading original IMDB dataset...")
    dataset = load_dataset("imdb")

    # 将原始训练集划分为训练集和验证集（例如 80% 训练，20% 验证）
    train_valid = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_valid["train"]
    val_dataset = train_valid["test"]
    test_dataset = dataset["test"]

    # 加载预训练 BERT 分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 定义分词函数，对文本进行编码
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",  # 填充至固定长度
            truncation=True,       # 超出最大长度时截断
            max_length=max_length  # 指定最大长度
        )

    # 对训练集、验证集、测试集进行批量分词处理
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 移除不需要的字段（原始文本字段）
    columns_to_remove = ["text"]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)
    test_dataset = test_dataset.remove_columns(columns_to_remove)

    # 设置格式为 PyTorch tensor 便于 DataLoader 使用
    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")

    # 如果 save 为 True，则将数据集保存到磁盘
    if save:
        print("Saving processed datasets to disk...")
        train_dataset.save_to_disk("train_dataset")
        val_dataset.save_to_disk("val_dataset")
        test_dataset.save_to_disk("test_dataset")

    return train_dataset, val_dataset, test_dataset, tokenizer

# 当直接运行 data.py 时，可测试数据加载与保存
if __name__ == "__main__":
    train_ds, val_ds, test_ds, tok = load_and_preprocess(save=True)
    print("Datasets saved!")
