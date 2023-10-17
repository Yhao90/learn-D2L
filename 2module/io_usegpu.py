import torch
from torch import nn
from torch.nn import functional as F

# 保存和读取张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')


# 保存模型的参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# !nvidia-smi 可以查看显卡信息
# 使用cpu和gpu，其中cuda和cuda:0等价
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
# 查询可用gpu个数
torch.cuda.device_count()


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpus()

# 默认情况下，张量是在CPU上创建的。
x = torch.tensor([1, 2, 3])
print(x.device)
# 使张量存放在gpu上
X = torch.ones(2, 3, device=try_gpu())
# 存在第2个gpu上，假设有多个
Y = torch.rand(2, 3, device=try_gpu(1))
# 把cpu张量复制到gpu1上，不在同一gpu或cpu无法运算,如果本来就在gpu1上，Z.cuda(1) is Z 返回Trye
X.cuda(1)

# 类似的，模型也能指定
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

# tip:不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算GPU上每个小批量的损失，并在命令行中将其报告给用户（或将其记录在NumPy ndarray中）时，
# 将触发全局解释器锁，从而使所有GPU阻塞。最好是为GPU内部的日志分配内存，并且只移动较大的日志。
