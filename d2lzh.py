import torch
import torchvision
import numpy as np
import sys
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
import os


def load_data_fashion_mnist(batch_size, resize=None, root='F:/SoftEnvironment/PycharmProjects/fashtion_mnist/data'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


# root='F:/SoftEnvironment/PycharmProjects/fashtion_mnist/data'
#
# trans = []
# trans.append(torchvision.transforms.ToTensor())
# transform = torchvision.transforms.Compose(trans)
#
# mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,transform=transform)
# mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,transform=transform)
#
#
# print(len(mnist_train))
# print(len(mnist_test))

# 下面的是测试的代码，不要删
# feature, label = mnist_train[0]
# ha0=feature.numpy()
# imsave('test.png',ha0[0,:,:])
#

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 设置为svg格式的
from IPython import display


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


#
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 这是打印前几个图形看看
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))


num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 想要去验证，取X的第一行X.view((-1, num_inputs))[0]，W不用取第一列兄弟
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def save_local(X, path_temp):
    file_name = os.path.join('F:/SoftEnvironment/PycharmProjects/fashtion_mnist/params/', path_temp)
    Q = np.array(X.detach().numpy())  # tensor转换成array
    np.savetxt(file_name, Q, fmt='%.04f')


# X：[256,784],  W:[784,10]  ,b：[10] ，这样得出的y_hat就是[256,10]

# 即256个样本，每一行10个类别对应着10个结果的概率（其中每一行加起来等于1）

# 在动手学深度学习中，有句话，交叉嫡只关心对正确类别的预测概率，因为只要其值足够大，就能保证分类是正确的

# y_hat.gather(1, y.view(-1, 1)) 是一个[256,1] 即存着正确分类的概率，那为什么要加上负号呢

# torch.log(torch.Tensor([0.0829]))=-2.4901
# torch.log(torch.Tensor([0.9]))=-0.1054  我们的目的是出现更多的0.9以上的概率（小于1）这样得出值更大
# 加上负号之后就是能让值更小了，还理解不清楚的话就是他们的作用都是离0更近了


num_epochs, lr = 5, 0.1


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    i=0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 看一下X的值
            save_local(X.view((-1, num_inputs)), 'X_' + str(i) + '.txt')
            # 看一下W初始值
            save_local(params[0], 'W_before' + str(i) + '.txt')
            # 看一下b初始值
            save_local(params[1], 'b_before' + str(i) + '.txt')
            y_hat = net(X)
            l = loss(y_hat, y).sum()  # l也有grad参数 和W b一样的

            # 梯度清零  第一次optimizer是none params[0].grad也是none，所以都不会进去
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 看一下W的梯度是多少
            save_local(params[0].grad, 'W_grad' + str(i) + '.txt')
            # 看一下b的梯度是多少
            save_local(params[1].grad, 'b_grad' + str(i) + '.txt')

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            # 看一下W更新参数之后的数值
            save_local(params[0], 'W_Step' + str(i) + '.txt')
            # 看一下b更新参数之后的数值
            save_local(params[1], 'b_Step' + str(i) + '.txt')
            i=i+1

            # 总的损失
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        # epoch 1, loss 0.7887, train acc 0.747, test acc 0.792
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
