# validate.py
# 此脚本用于验证训练好的BERT情感分析模型。
# 用户运行程序后，可手动输入一句话，程序将调用训练好的模型进行预测并输出情绪类别。

from transformers import BertForSequenceClassification, BertTokenizer
import torch


def predict_emotion(text, model, tokenizer, device):
    """
    对输入文本进行情感预测
    参数:
        text: 用户输入的文本字符串
        model: 加载好的BERT情感分类模型
        tokenizer: 对应的分词器
        device: 模型所在设备（GPU或CPU）
    返回:
        pred_label: 模型预测的标签，0表示负面，1表示正面
    """
    # 使用分词器对输入文本进行编码，设置固定最大长度为128，自动填充和截断
    inputs = tokenizer(
        text,
        padding="max_length",  # 填充至最大长度
        truncation=True,  # 超出最大长度时截断
        max_length=128,  # 最大长度设置为128
        return_tensors="pt"  # 返回PyTorch tensor格式
    )

    # 将编码后的输入数据移动到指定设备（GPU或CPU）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 将模型设置为评估模式，关闭dropout等
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果（取logits中最大值对应的下标）
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    return pred_label


def main():
    # 指定模型和分词器保存路径，确保路径与训练时保存路径一致
    model_save_path = "bert_imdb_finetuned"  # 如有时间戳，请替换为对应目录

    # 加载训练好的BERT模型和分词器
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # 配置设备：如果GPU可用，则使用GPU，否则使用CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # 提示用户输入一段文本
    input_text = input("请输入一句评论：")

    # 调用预测函数，获取预测标签
    pred = predict_emotion(input_text, model, tokenizer, device)

    # 根据标签输出情感类别，假设1表示正面，0表示负面
    if pred == 1:
        print("预测情感为：正面")
    else:
        print("预测情感为：负面")


if __name__ == "__main__":
    main()
