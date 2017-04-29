## Vision functions
### torch.nn.functional.pixel_shuffle(input, upscale_factor)[source]

将形状为`[*, C*r^2, H, W]`的`Tensor`重新排列成形状为`[C, H*r, W*r]`的Tensor.

详细请看[PixelShuffle](..).

形参说明:
- input (Variable) – 输入
- upscale_factor (int) – 增加空间分辨率的因子.

例子:
```python
ps = nn.PixelShuffle(3)
input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
output = ps(input)
print(output.size())
torch.Size([1, 1, 12, 12])
```
### torch.nn.functional.pad(input, pad, mode='constant', value=0)[source]

填充`Tensor`.

目前为止,只支持`2D`和`3D`填充.
Currently only 2D and 3D padding supported.
当输入为`4D Tensor`的时候,`pad`应该是一个4元素的`tuple (pad_l, pad_r, pad_t, pad_b )` ,当输入为`5D Tensor`的时候,`pad`应该是一个6元素的`tuple (pleft, pright, ptop, pbottom, pfront, pback)`.

形参说明:
input (Variable) – 4D 或 5D `tensor`
pad (tuple) – 4元素 或 6-元素  `tuple`
mode – ‘constant’, ‘reflect’ or ‘replicate’
value – 用于`constant padding` 的值.
