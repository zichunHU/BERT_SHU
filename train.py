# train.py
import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    在训练集上执行一个训练轮次

    参数:
        model: 待训练的模型
        dataloader: 训练数据的 DataLoader
        optimizer: 用于更新模型参数的优化器
        device: 模型和数据所在设备（GPU 或 CPU）
        scheduler: 学习率调度器（可选），用于动态调整学习率
    返回:
        avg_loss: 当前轮次的平均训练损失
    """
    model.train()  # 切换到训练模式
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # 将输入数据移动到指定设备上
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # 清除上一步梯度

        # 前向传播并计算损失
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

        # 如果传入了 scheduler，则更新学习率
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    """
    在验证集上评估模型性能，计算平均损失、准确率、精确率、召回率和 F1 分数

    参数:
        model: 待评估的模型
        dataloader: 验证数据的 DataLoader
        device: 数据所在设备（GPU 或 CPU）
    返回:
        avg_loss: 平均验证损失
        accuracy: 验证准确率
        precision: 精确率
        recall: 召回率
        f1: F1 分数
    """
    model.eval()  # 切换到评估模式
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # 预测：取 logits 中概率最大的类别
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    print("请通过 main.py 调用训练和评估函数。")
