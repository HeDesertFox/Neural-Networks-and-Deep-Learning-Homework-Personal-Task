import numpy as np
import matplotlib.pyplot as plt
from data_handling import batch_generator

# Loss Functions
def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # 防止 log(0)
    return loss

def delta_cross_entropy(y_pred, y_true):
    return y_pred - y_true  # 这里 y_pred 是 softmax 的输出


# Optimizer & Regularization
class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        # 获取模型的参数及其梯度
        for param, grad in self.model.get_params_and_grads():
            # 应用梯度下降
            param -= self.lr * (grad + self.weight_decay * param)

    # def update_parameters(self, new_params):
    #     # 直接用新的参数值覆盖旧的参数
    #     # new_params 应该是一个和 self.parameters 同结构的列表，其中包含新的参数值
    #     if len(new_params) != len(self.parameters):
    #         raise ValueError("New parameters list must match the original in length.")

    #     for (param, _), new_param in zip(self.parameters, new_params):
    #         # 确保新参数的形状与旧参数相同
    #         if param.shape != new_param.shape:
    #             raise ValueError("New parameter shapes must match the original shapes.")
    #         # 直接覆盖参数
    #         np.copyto(param, new_param)


# Scheduler
class LRScheduler:
    def __init__(self, optimizer, decay_rate, step_size):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.step_size == 0:
            self.optimizer.lr *= self.decay_rate

# Evaluate Function
def accuracy(y_pred, y_true):
    """
    计算准确率
    :param y_pred: 模型的预测结果，维度是(batch_size, num_classes)，每一行是一个样本的预测概率
    :param y_true: 真实标签，维度是(batch_size, num_classes)，每一行是一个样本的真实标签（假设已经是one-hot编码）
    :return: 准确率
    """
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy


def evaluate(model, data_loader):
    model.eval()  # Switch model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        y_pred = model.forward(x_batch)
        loss = cross_entropy_loss(y_pred, y_batch)
        total_loss += loss * x_batch.shape[0]
        acc = accuracy(y_pred, y_batch)  # Assuming accuracy function is defined elsewhere
        total_accuracy += acc * x_batch.shape[0]
        total_samples += x_batch.shape[0]

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy

# Training Loop
def train_and_evaluate(model, X_train, Y_train, X_val, Y_val, optimizer, scheduler, batch_size=64, epochs=10):
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        # 每个epoch开始时创建新的数据加载器
        train_loader = batch_generator(X_train, Y_train, batch_size=batch_size)
        val_loader = batch_generator(X_val, Y_val, batch_size=batch_size)

        model.train()
        train_loss = 0
        total_batches = 0  # 用于计数总的批次数
        total_accuracy = 0  # 用于累计正确率

        for x_batch, y_batch in train_loader:
            total_batches += 1  # 对每个批次进行计数
            y_pred = model.forward(x_batch)
            loss = cross_entropy_loss(y_pred, y_batch)
            train_loss += loss

            # 计算准确率
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            total_accuracy += accuracy

            model.zero_grad()
            grad_output = delta_cross_entropy(y_pred, y_batch)
            model.backward(grad_output)
            optimizer.step()

        # 计算平均训练损失和平均训练准确率
        train_loss /= total_batches if total_batches > 0 else 1  # 防止除以零
        train_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        # 在验证集上评估模型
        val_loss, val_accuracy = evaluate(model, val_loader)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)

        # 调用scheduler的step方法来更新学习率
        scheduler.step()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return history


def plot_training_history(history):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 绘制准确率曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='tab:blue')
    # ax1.plot(history['train_accuracy'], label='Train Accuracy', color='tab:orange', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='lower left')

    # 创建一个共享X轴的次Y轴用于损失
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(history['train_loss'], label='Train Loss', color='tab:red')
    ax2.plot(history['val_loss'], label='Validation Loss', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper left')

    fig.tight_layout()  # 调整布局以防止重叠
    plt.title('Training and Validation Metrics')
    plt.show()
