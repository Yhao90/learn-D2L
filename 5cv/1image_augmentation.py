import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

# 图像增广，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
# d2l.plt.imshow(img);
plt.imshow(img)


# plt.show()

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)


apply(img, torchvision.transforms.RandomHorizontalFlip())

# 水平翻转
torchvision.transforms.RandomHorizontalFlip()
# 垂直翻转，注意部分场景不可用，例如图片是山，房子
torchvision.transforms.RandomVerticalFlip()
# 随机剪裁,剪裁一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# 改变颜色,改变图像颜色的四个方面：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
# brightness=0.5即亮度为原来的50%-150%
torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

train_with_data_aug(train_augs, test_augs, net)