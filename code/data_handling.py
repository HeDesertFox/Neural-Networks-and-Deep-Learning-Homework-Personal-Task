import gzip
import numpy as np
import urllib.request
import os

def download(url, file_name):
    """下载数据集文件，如果文件不存在的话。"""
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)

def load_mnist_images(file_name):
    """加载MNIST图像数据，并进行归一化和展平。"""
    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)  # 28x28图像展平成784向量
    return data / np.float32(256)  # 归一化

def load_mnist_labels(file_name):
    """加载MNIST标签数据，并转换为one-hot编码。"""
    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return np.eye(10)[data]  # 转换为one-hot编码

def split_train_val(X, Y, val_fraction=0.1):
    """从训练数据中随机划分出验证集。"""
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    total_samples = X.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    val_size = int(total_samples * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return X[train_indices], Y[train_indices], X[val_indices], Y[val_indices]

def batch_generator(X, Y, batch_size=64, shuffle=True):
    """生成批次数据的生成器。如果是测试集，可以将batch_size设置为None来处理整个数据集。"""
    if batch_size is None:
        yield X, Y
    else:
        n_batches = len(X) // batch_size
        if shuffle:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
        for i in range(n_batches):
            begin = i * batch_size
            end = begin + batch_size
            yield X[begin:end], Y[begin:end]

        if len(X) % batch_size != 0:
            yield X[n_batches*batch_size:len(X)], Y[n_batches*batch_size:len(Y)]
