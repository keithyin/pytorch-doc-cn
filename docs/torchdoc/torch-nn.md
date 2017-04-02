# torch.nn

## Parameters
### class torch.nn.Parameter()
`Variable`的一种，常被用于模块参数(`module parameter`)。

`Parameters` 是 `Variable` 的子类。`Paramenters`和`Modules`一起使用的时候会有一些特殊的属性，即：当`Paramenters`赋值给`Module`的属性的时候，他会自动的被加到 `Module`的 参数列表中(即：会出现在 `parameters() 迭代器中`)。将`Varibale`赋值给`Module`属性则不会有这样的影响。
这样做的原因是：我们有时候会需要缓存一些临时的状态(`state`), 比如：模型中`RNN`的最后一个隐状态。如果没有`Parameter`这个类的话，那么这些临时变量也会注册成为模型变量。

`Variable` 与 `Parameter`的另一个不同之处在于，`Parameter`不能被 `volatile`(即：无法设置`volatile=True`)而且默认`requires_grad=True`。`Variable`默认`requires_grad=False`。


参数说明:

- data (Tensor) – parameter tensor.

- requires_grad (bool, optional) – 默认为`True`，在`BP`的过程中会对其求微分。

## Containers（容器）：
### class torch.nn.Module
所有神经网络的基类。

你的模型也应该继承这个类。

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
Submodules assigned in this way will be registered, and will have their parameters converted too when you call .cuda(), etc.

add_module(name, module)[source]
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.

children()[source]
Returns an iterator over immediate children modules.

cpu(device_id=None)[source]
Moves all model parameters and buffers to the CPU.

cuda(device_id=None)[source]
Moves all model parameters and buffers to the GPU.

Parameters:	device_id (int, optional) – if specified, all parameters will be copied to that device
double()[source]
Casts all parameters and buffers to double datatype.

eval()[source]
Sets the module in evaluation mode.

This has any effect only on modules such as Dropout or BatchNorm.

float()[source]
Casts all parameters and buffers to float datatype.

forward(*input)[source]
Defines the computation performed at every call.

Should be overriden by all subclasses.

half()[source]
Casts all parameters and buffers to half datatype.

load_state_dict(state_dict)[source]
Copies parameters and buffers from state_dict into this module and its descendants. The keys of state_dict must exactly match the keys returned by this module’s state_dict() function.

Parameters:	state_dict (dict) – A dict containing parameters and persistent buffers.
modules()[source]
Returns an iterator over all modules in the network.
