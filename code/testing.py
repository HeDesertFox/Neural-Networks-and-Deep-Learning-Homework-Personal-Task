import numpy as np

# Evaluation Metrics
def accuracy(y_pred, y_true):
    """
    计算准确率
    :param y_pred: 模型的预测结果，维度是(batch_size, num_classes)，每一行是一个样本的预测概率
    :param y_true: 真实标签，维度是(batch_size, num_classes)，每一行是一个样本的真实标签
    :return: 准确率
    """
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

# Testing Loop
def test(model, test_loader):
    """
    在测试集上评估模型性能
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器，可以是一次性加载整个测试集，也可以是批量加载
    :return: 测试集上的平均准确率
    """
    model.eval()  # 设置模型为评估模式
    total_accuracy = 0
    total_samples = 0

    for x_batch, y_batch in test_loader:
        y_pred = model.forward(x_batch)
        total_accuracy += accuracy(y_pred, y_batch) * x_batch.shape[0]
        total_samples += x_batch.shape[0]

    avg_accuracy = total_accuracy / total_samples
    return avg_accuracy
