import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 在目标检测中，我们通常使用边界框（bounding box,bbox）来描述对象的空间位置。
# 边界框是矩形的，通常由矩形左上角的以及右下角的和坐标决定。

d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
fig = d2l.plt.imshow(img)

# dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
# fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
# fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))
# plt.show()

# 精简输出精度
torch.set_printoptions(2)

h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = d2l.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)  # torch.Size([1, 2042040, 4]) ,（批量大小，锚框的数量，4）
# 将锚框变量Y的形状更改为(图像高度,图像宽度,以同一像素为中心的锚框的数量,4)后
boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250, 0, :])  # 访问以（250,250）为中心的第一个锚框
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
# 变量boxes中轴和轴的坐标值已分别除以图像的宽度和高度。
# 绘制锚框时，我们需要恢复它们原始的坐标值。 因此，在下面定义了变量bbox_scale
d2l.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
plt.show()
