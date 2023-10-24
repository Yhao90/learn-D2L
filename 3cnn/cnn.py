import torch
from torch import nn
from d2l import torch as d2l

'''
    平移不变性（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
    局部性（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。
'''


# 输入大小 nℎ×n𝑤 ,核大小 kℎ×k𝑤 ,步长1的时候输出大小(nℎ−kℎ+1)×(n𝑤−k𝑤+1)
# 步幅为 Sℎ×S𝑤 ,( (nℎ−kℎ+Sℎ)/Sℎ )×( (n𝑤−k𝑤+S𝑤)/S𝑤 )
def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 简单卷积层实现,我们将带有ℎ×𝑤卷积核的卷积层称为ℎ×𝑤卷积层。
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 小case，图像的垂直边缘检测,1白色，0黑色
X = torch.ones((6, 8))  # 图像
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])  # 卷积核
Y = corr2d(X, K)  # 输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘

# 学习卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率
# 所学的卷积核的权重张量 conv2d.weight.data.reshape((1, 2)) 最后结果和上面的K很接近
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列,输入通道1，输出通道1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # stride=2 步幅为2
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape  # 输出依然是(8,8)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))  # pad 高宽，步幅 高宽


# 多输入通道，单输出通道
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


# 多输入多输出
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 1x1卷积核，其实相当于全连接层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


# 池化层，我们通常希望这些特征保持某种程度上的平移不变性,降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。
# 分为最大池化层和平均池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# 3x3池化层，填充1，步幅2。对于多通道，是每个通道和对应的池化层计算，不像卷积一样全部计算再汇总
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
