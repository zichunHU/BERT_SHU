# main.py
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from data import load_and_preprocess
from model import build_model
from train import train_epoch, evaluate


def main():
    # -------------------------
    # 1. 数据加载与预处理
    # -------------------------
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess(
        max_length=128,
        test_size=0.2,
        seed=42,
        save=False,
        load_saved=True  # 如果之前已保存，可设置为 True
    )

    # -------------------------
    # 2. 构建 DataLoader
    # -------------------------
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 3. 构建模型
    # -------------------------
    model = build_model(num_labels=2)

    # -------------------------
    # 4. 配置设备（GPU 或 CPU）
    # -------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # -------------------------
    # 5. 设置优化器（固定学习率）
    # -------------------------
    learning_rate = 2e-5
    weight_decay = 0.01
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 10

    # -------------------------
    # 6. 设置 TensorBoard 日志记录
    # -------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/experiment_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to {log_dir}")

    # -------------------------
    # 7. 训练与验证循环
    # -------------------------
    for epoch in range(epochs):
        print(f"\n----- Epoch {epoch + 1} -----")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        # 固定学习率，不会变化，所以不需要记录学习率更新

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)

    # -------------------------
    # 8. 保存模型和分词器
    # -------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_save_path = f"bert_imdb_finetuned_{timestamp}"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved at {model_save_path}")

    writer.close()


if __name__ == "__main__":
    main()
