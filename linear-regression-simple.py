import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

# 线性回归简单实现，使用pytorch处理数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成数据
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据 is_train 是否打乱数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# next(iter(data_iter)) # iter转为迭代器

# 定义模型，线性回归模型，(2,1)表示输入2维，输出1维，Sequential 相当于容器 list of layers
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数，w即weight 均值0，方差0.01，bias即b偏差，设置为0
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 定义损失函数，均方误差
loss = nn.MSELoss()

# 实例化sgd实例,即优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
