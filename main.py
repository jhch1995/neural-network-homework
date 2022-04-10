#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
* This file is the first homework of neural network and deep learning
* To achieve a simple classification task by using two layer neural network.
* Thanks to d2l course supporting for open source code contributions.
* Copyright (C) 2022 Haochen Jiang <hcjiang21@m.fudan.edu.cn> (School of DataScience in Fudan University)
"""

import torchvision
import numpy as np
from plot_image import *
from torch.utils import data
from torchvision import transforms
import sys
import argparse

class Relu(object):
    def __init__(self, X):
        self.x = None
        self.forward(X)

    @staticmethod
    def forward(self, X):
        self.x = X
        return np.maximum(X, 0)

    def backward(self, gradient):
        gradient[self.x <= 0] = 0
        return gradient

def sigmoid(X):
    return 1./(1+np.exp(-X))

def sigmoid_grad(X):
    return (1.0 - sigmoid(X)) * sigmoid(X)

def softmax(X):
    # pass
    output = np.exp(X - X.max(axis=-1, keepdims=True))
    return output / output.sum(axis=-1, keepdims=True)

class CrossEntropyLoss(object):
    def __init__(self, X, y, class_count):
        self.forward(X, y, class_count)

    @staticmethod
    def forward(predict, y, class_count):
        one_hot_label = gen_one_hot_label(y, class_count)
        tmp = np.log(predict) * one_hot_label
        # sum of one batch
        batch_size = y.shape[0]
        loss = -np.sum(tmp + 1e-7) / batch_size
        return loss

    @staticmethod
    def backward(self, predict, y):
        return predict - y
        pass

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

def gen_one_hot_label(y, class_count):
    return np.eye(class_count)[y, :]

def l2_regularizer(W):
    l2_reg_square = W ** 2
    return np.sum(l2_reg_square)


class Net:
    def __init__(self, input_size, hidden_size, output_size):
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

    def backprop(self, X, y_hat):
        grads = {}
        batch_num = X.shape[0]
        X = X.reshape((-1, self.input_size))
        o1 = np.dot(X, self.weight1) + self.bias1
        o2 = sigmoid(o1)
        o3 = np.dot(o2, self.weight2) + self.bias2
        y = softmax(o3)

        y_hat = gen_one_hot_label(y_hat, self.output_size)

        dy = (y - y_hat) / batch_num
        # dy 32 * 10
        grads['weight2'] = np.dot(o2.T, dy)
        # print(f"grad weight2 shape: {np.dot(o2.T, dy).shape}")
        # 在第一个维度上求和剩下列维度，
        grads['bias2'] = np.sum(dy, axis=0)
        # print(f"grad bias2 shape: {np.sum(dy, axis=0).shape}")

        do2 = np.dot(dy, self.weight2.T)
        do1 = sigmoid_grad(o1) * do2
        grads['weight1'] = np.dot(X.T, do1)
        # print(f"grad weight1 shape: {np.dot(X.T, do1).shape}")
        grads['bias1'] = np.sum(do1, axis=0)

        return grads

    def loss(self, X, y_hat):
        y = self.forward(X)
        loss = CrossEntropyLoss.forward(y, y_hat, self.output_size)
        return loss

    def accuracy(self, X, y_hat):
        y = self.forward(X)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == y_hat) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self):
        pass

    def param_plot(self):
        W1_plot = self.weight1
        print("self.weight1 shape", self.weight1.shape)
        W1_plot = W1_plot.reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
        print("W1_plot", W1_plot.shape)
        plt.figure(figsize=(5, 5))
        for index in range(W1_plot.shape[0]):
            plt.subplot(2, 5, index + 1)
            plt.imshow(W1_plot[index, :, :], cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('./plot/one_layer_parameterW1.png')

def train_epoch(net, train_iter, lr, alpha):
    metric = Accumulator(3)
    regular_alpha = alpha
    for X, y_hat in train_iter:
        X = X.numpy()
        y_hat = y_hat.numpy()
        loss = net.loss(X, y_hat) + alpha * (l2_regularizer(net.weight1) + l2_regularizer(net.weight2))
        grad = net.backprop(X, y_hat)
        net.weight2 -= lr * (grad['weight2'] + 2 * regular_alpha * net.weight2)
        net.bias2 -= lr * grad['bias2']
        net.weight1 -= lr * (grad['weight1'] + 2 * regular_alpha * net.weight1)
        net.bias1 -= lr * grad['bias1']
        # 这块可能要调一下
        # print(f"train loss {loss}")
        # print(f"net accuracy: {net.accuracy(X, y_hat)}")
        metric.add(loss, net.accuracy(X, y_hat), 1)
    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y_hat in data_iter:
        X = X.numpy()
        y_hat = y_hat.numpy()
        # 以batch为单位的
        metric.add(net.accuracy(X, y_hat), 1)
    return metric[0] / metric[1]

def save_modules_parameter(net, save_path, file_name = 'jhc_module_param'):
    params = {}
    params['weight2'] = net.weight2
    params['weight1'] = net.weight1
    params['bias1'] = net.bias1
    params['bias2'] = net.bias2
    np.save(save_path + file_name, params)

def train(net, train_iter, test_iter, lr, alpha, num_epochs):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        if (epoch) % 3 == 0 and epoch != 0:
            lr = lr * 1
        train_metrics = train_epoch(net, train_iter, lr, alpha)
        print(f"train epoch loss {train_metrics[0]}")
        print(f"net epoch accuracy: {train_metrics[1]}")
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"net epoch test accuracy: {test_acc}")
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def train_without_plot(net, train_iter, test_iter, lr, alpha, num_epochs):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, lr, alpha)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    return test_acc

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # data loader
    num_inputs = 784
    num_outputs = 10
    hidden_size = 100
    regular_param = 0
    learning_rate = 0.1
    MLP_Net = Net(num_inputs, hidden_size, num_outputs)
    # batch: 32
    # dataset 6000 * 28 * 28
    # 28*28 = 784 = num_imputs
    train_iter, test_iter = load_data_mnist(args.batch_size)

    # train_epoch(MLP_Net, train_iter, 0.01, 2)
    train(MLP_Net, train_iter, test_iter, learning_rate, regular_param, args.train_num_epochs)

    # save parameter
    save_module_path = './model_param/'
    save_modules_parameter(MLP_Net, save_module_path)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
