import numpy as np
import matplotlib.pyplot as plt
import os

class Linear:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim) # He初始化
        self.bias = np.zeros((1, output_dim))
        self.input = None
        # Initialize gradients as None, will be updated during backprop
        self.gradW = np.zeros_like(self.weights)
        self.gradB = np.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        self.gradW = np.dot(self.input.T, grad_output)
        self.gradB = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 提升数值稳定性
    return e_x / e_x.sum(axis=1, keepdims=True)

class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation_fn=ReLU):
        self.layer1 = Linear(input_dim, hidden_dim1)
        self.activation1 = activation_fn()
        self.layer2 = Linear(hidden_dim1, hidden_dim2)
        self.activation2 = activation_fn()
        self.layer3 = Linear(hidden_dim2, output_dim)
        # Store layers in a list for easier access to all layers' parameters and gradients
        self.layers = [self.layer1, self.activation1, self.layer2, self.activation2, self.layer3]

    def forward(self, x):
        for layer in self.layers[:-1]:  # Exclude the last layer which is output layer
            x = layer.forward(x)
        x = self.layers[-1].forward(x)  # Output layer
        return softmax(x)  # Apply softmax to the output of the last layer

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_params_and_grads(self):
        # Aggregate parameters and their gradients from all layers that have parameters
        params_and_grads = []
        for layer in self.layers:
            if isinstance(layer, Linear):  # Check if layer has parameters
                params_and_grads.append((layer.weights, layer.gradW))
                params_and_grads.append((layer.bias, layer.gradB))
        return params_and_grads

    def zero_grad(self):
        # Clear gradients of all parameters
        for layer in self.layers:
            if isinstance(layer, Linear):  # Check if layer has parameters
                layer.gradW = np.zeros_like(layer.weights)
                layer.gradB = np.zeros_like(layer.bias)


    def train(self):
        # 对于更复杂的模型，这里可以用来启用训练特定的层行为，如Dropout
        pass

    def eval(self):
        # 对于更复杂的模型，这里可以用来禁用训练特定的层行为，如Dropout
        pass

# 可视化
def visualize_weights(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            plt.figure(figsize=(10, 5))
            plt.imshow(layer.weights.T, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Layer {i+1} Weight Heatmap')
            plt.xlabel('Output Nodes')
            plt.ylabel('Input Nodes')
            plt.show()

# 保存或导入模型权重
def save_model_weights(model, filename):
    data_to_save = {}
    # 收集所有参数和其梯度
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):  # 确保只处理有权重和偏差的层
            data_to_save[f'weights_{i}'] = layer.weights
            data_to_save[f'bias_{i}'] = layer.bias

    # 保存到文件
    np.savez(filename, **data_to_save)
    print(f"Model weights saved to {filename}")

def load_model_weights(model, filename):
    data = np.load(filename)
    layer_index = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            # 直接将 numpy 数组赋值给模型层的权重和偏置
            layer.weights = data[f'weights_{i}']
            layer.bias = data[f'bias_{i}']
            layer_index += 1
    print(f"Model weights loaded from {filename}")