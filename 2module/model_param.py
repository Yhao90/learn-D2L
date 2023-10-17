import torch
from torch import nn
from torch.nn import functional as F


# 自定义块

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


X = torch.rand(2, 20)
net = MLP()
out = net(X)
print(out)


# 顺序块，自定义Sequential
class MySequential(nn.Module):
    # 每个Module都有_modules，主要优点是： 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)


# 混合使用
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


net = nn.Sequential(NestMLP(), nn.Linear(16, 20), MLP())
net(X)

# param管理
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# 访问参数,检查第二个全连接层的参数。
# OrderedDict([('weight',
# tensor([[-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, -0.0334, -0.2263]])), ('bias', tensor([0.0887]))])
print(net[2].state_dict())
# tensor([0.0887], requires_grad=True)
# tensor([0.0887])
# True
print(net[2].bias)
print(net[2].bias.data)
print(net.state_dict()['2.bias'].data)
print(net[2].weight.grad == None)

# 遍历所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 嵌套访问，先print模型，看结构，访问，例如 rgnet[0][1][0].bias.data

# 参数初始化，默认是pytorch会根据一个范围均匀地初始化权重和偏置矩阵， 这个范围是根据输入和输出维度计算出的。
# PyTorch的nn.init模块提供了多种预置初始化方法。

'''
均匀分布,从均匀分布U ( a , b ) U(a,b)U(a,b)中生成值，填充输入的张量或变量。
torch.nn.init.uniform_(tensor, a=0.0, b=1.0) 
正态分布
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
对角分布
torch.nn.init.eye_(tensor)
dirac分布
torch.nn.init.dirac_(tensor, groups=1)
xavier_uniform分布,gain可选的缩放因子
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
xavier_normal分布
torch.nn.init.xavier_normal_(tensor, gain=1.0)
kaiming_uniform 分布
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
kaiming_normal分布
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
正交分布
torch.nn.init.orthogonal_(tensor, gain=1)
稀疏矩阵
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
'''


def init_normal(m):
    if type(m) == nn.Linear:
        # 将所有权重参数初始化为标准差为0.01的高斯随机变量
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # 偏移都为0
        nn.init.zeros_(m.bias)
        # 设置为常量 1
        # nn.init.constant_(m.weight, 1)
        # nn.init.xavier_uniform_(m.weight)


# 应用初始化的参数，单指定某层 net[0].apply(init_xavier)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
# 直接设置参数
net[0].weight.data[0, 0] = 42
net[0].weight.data[:] += 1

# 共享参数，设置一个共享层
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])  # True
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

# 延迟初始化
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# print(net[0].weight)  # 尚未初始化
print(net)

X = torch.rand(2, 20)
net(X)
print(net)
net = nn.Sequential(
    nn.Linear(20, 256), nn.ReLU(),  # 立即初始化
    nn.LazyLinear(128), nn.ReLU(),  # 延迟初始化
    nn.LazyLinear(10)
)
