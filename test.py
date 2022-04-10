"""
* This file is a part of the first homework of neural network and deep learning
* To achieve a simple classification task by using two layer neural network.
* Thanks to d2l course supporting for open source code contributions.
* Copyright (C) 2022 Haochen Jiang <hcjiang21@m.fudan.edu.cn> (School of DataScience in Fudan University)
"""

import torchvision
import numpy as np
from plot_image import *
from torch.utils import data
from torchvision import transforms

def load_modules_parameter(net, load_path, file_name = 'jhc_module_param.npy'):
    # mmap_mode : {None, ‘r+’, ‘r’, ‘w+’, ‘c’}, optional
    params = np.load(load_path + file_name, allow_pickle=True).item()
    net.weight2 = params['weight2']
    net.weight1 = params['weight1']
    net.bias2 = params['bias2']
    net.bias1 = params['bias1']

def sigmoid(X):
    return 1./(1+np.exp(-X))

def sigmoid_grad(X):
    return (1.0 - sigmoid(X)) * sigmoid(X)

def softmax(X):
    # pass
    output = np.exp(X - X.max(axis=-1, keepdims=True))
    return output / output.sum(axis=-1, keepdims=True)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Net:
    def __init__(self, input_size = 784, hidden_size = 50 , output_size = 10):
        # easy for parameter tunning
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight1 = np.random.normal(0, 0.01, (input_size, hidden_size))
        # broad cast
        self.bias1 = np.zeros(hidden_size)
        self.weight2 = np.random.normal(0, 0.01, (hidden_size, output_size))
        self.bias2 = np.zeros(output_size)

    def forward(self, X):
        # it seems to use a list
        # reshape can merge the c, h, w to num_inputs
        # input X : 32 * 1 * 28 * 28
        X = X.reshape((-1, self.input_size))
        layer1_out = sigmoid(np.dot(X, self.weight1) + self.bias1)
        layer2_out = softmax(np.dot(layer1_out, self.weight2) + self.bias2)
        return layer2_out

    def accuracy(self, X, y_hat):
        y = self.forward(X)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == y_hat) / float(X.shape[0])
        return accuracy

    def param_plot(self):
        W1_plot = self.weight2
        print("self.weight1 shape", self.weight1.shape)
        W1_plot = W1_plot.reshape(10, 10, 1, -1).transpose(3, 0, 1, 2)
        print("W1_plot", W1_plot.shape)
        plt.figure(figsize=(5, 5))
        for index in range(W1_plot.shape[0]):
            plt.subplot(2, 5, index+1)
            plt.imshow(W1_plot[index, :, :], cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('./plot/MLP_W2_vis.png')

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y_hat in data_iter:
        X = X.numpy()
        y_hat = y_hat.numpy()
        # 以batch为单位的
        metric.add(net.accuracy(X, y_hat), 1)
    return metric[0] / metric[1]

def load_data_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    # compose img transforms operation
    trans = transforms.Compose(trans)
    # mnist train and test data load
    mnist_train = torchvision.datasets.MNIST(root='../data', train=True, transform=trans, target_transform=None,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST(root='../data', train=False, transform=trans, target_transform=None,
                                            download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

def test_accuracy(net, test_iter, num_epochs):
    test_acc = evaluate_accuracy(net, test_iter)

if __name__ == '__main__':
    MLP_Net = Net()
    load_path = '/home/fudan-ads/homework/nndlpj1/model_param/'
    load_modules_parameter(MLP_Net, load_path)

    train_iter, test_iter = load_data_mnist(32)

    test_acc = evaluate_accuracy(MLP_Net, test_iter)

    print(f"current dataset accuracy is:  {test_acc}")
    MLP_Net.param_plot()



