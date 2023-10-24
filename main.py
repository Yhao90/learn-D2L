import torch

import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)  # X.shape (8,8)->(1,1,8,8)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# print(comp_conv2d(conv2d, X).shape)  # (8,8)

# X = torch.ones(2, 2)
# a = torch.rand(3)
# print(a)  # tensor([0.4609, 0.4397, 0.7345])
# print(a > 0.5)  # tensor([False, False,  True])
# b = (a > 0.5).float()
# print(b)  # tensor([0., 0., 1.])

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
for x, k in zip(X, K):
    print(x)
    print(k)
