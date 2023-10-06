import torch
from IPython import display
from d2l import torch as d2l

# softmax回归的从零开始实现,虽然称为回归但是是用在分类问题上
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
