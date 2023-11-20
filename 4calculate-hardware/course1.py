import torch
from torch import nn
from d2l import torch as d2l

# 普通的代码是命令式编程，以下可以转变为符号式编程
'''
命令式编程更容易使用。在Python中，命令式编程的大部分代码都是简单易懂的。
命令式编程也更容易调试，这是因为无论是获取和打印所有的中间变量值，或者使用Python的内置调试工具都更加简单；

符号式编程运行效率更高，更易于移植。符号式编程更容易在编译期间优化代码，同时还能够将程序移植到与Python无关的格式中，
从而允许程序在非Python环境中运行，避免了任何潜在的与Python解释器相关的性能问题。
'''


def add_():
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''


def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'


prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)


# 以下转换成符号式编程
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2))
    return net


net = get_net()
net = torch.jit.script(net)

# 默认情况下，GPU操作在PyTorch中是异步的，计算由后端gpu执行，而前端将控制权返回给了Python。
# 如果要等到gpu计算完可以使用api  torch.cuda.synchronize(device)
# 后端gpu执行的时候是c++代码，因此不管python性能如何，都不大会影响程序的效率

# 指定gpu计算
devices = d2l.try_all_gpus()


def run(x):
    return [x.mm(x) for _ in range(50)]


x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
run(x_gpu1)
run(x_gpu2)  # 预热设备
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])
with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])
with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])

# 不指定具体gpu
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
# 在上述情况下，总执行时间小于两个部分执行时间的总和，因为深度学习框架自动调度两个GPU设备上的计算，而不需要用户编写复杂的代码。

