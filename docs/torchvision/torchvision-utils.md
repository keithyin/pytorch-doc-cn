# torchvision.utils

## torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False)
猜测，用来做 `雪碧图的`（`sprite image`）。

给定 `4D mini-batch Tensor`， 形状为 `(B x C x H x W)`,或者一个`a list of image Tensor`，做成一个`size`为`(B / nrow, nrow)`的雪碧图。

- normalize=True ，会将图片的像素值归一化处理

- 如果 range=(min, max)， min和max是数字，那么`min`，`max`用来规范化`image`

- scale_each=True ，每个图片独立规范化，而不是根据所有图片的像素最大最小值来规范化

- padding: 小图之间的间隔

`NOTE:`
如果使用的是`python3x`，如果下面代码跑不起来的话，就把报错地方的源码 `long`改成`int`。
还有一个需要注意的是：`Channel`不能为`4`，是`4`的话会报错，`3`可以正常运行。

例子：

```python
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

lena = scipy.misc.face()
img = transforms.ToTensor()(lena)
print(img.size())
imglist = [img, img, img, img.clone().fill_(-10)]
show(make_grid(imglist, nrow=2, padding=5))
plt.show()
```

[Example usage is given in this notebook](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)



## torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False)

将给定的`Tensor`保存成image文件。如果给定的是`mini-batch tensor`，那就用`make-grid`做成雪碧图，再保存。
