import torch

X = torch.ones(2,2)
a = torch.rand(3)
print(a)
print(a > 0.5)
b = (a > 0.5).float()
print(b)
