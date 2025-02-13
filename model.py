# model.py
# 本模块加载预训练的 BERT 模型，并在其基础上构建序列分类模型，
# 其中情感分析任务为二分类问题（正面、负面）。

from transformers import BertForSequenceClassification

def build_model(num_labels=2):
    """
    构建用于序列分类的 BERT 模型
    参数:
        num_labels: 分类任务的类别数，此处二分类问题 num_labels=2
    返回:
        model: 加载好的 BertForSequenceClassification 模型
    """
    # 加载预训练的 bert-base-uncased 模型，并添加分类层
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    return model

# 测试模型加载
if __name__ == "__main__":
    model = build_model()
    print(model)
