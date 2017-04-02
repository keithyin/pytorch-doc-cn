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

`Modules`也可以包含其它`Modules`,允许使用树结构嵌入他们。你可以将子模块赋值给模型属性。
```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)# submodule: Conv2d
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
```

通过上面方式赋值的`submodule`会被注册。当调用 `.cuda()` 的时候，`submodule`的参数也会转换为`cuda Tensor`。

#### add_module(name, module)
将一个 `child module` 添加到当前 `modle`。
被添加的`module`可以通过 `name`属性来获取。
例：
```python
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
model = Model()
print(model.conv)
```
输出：
```
Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
```

#### children()
Returns an iterator over immediate children modules.
返回当前模型 子模块的迭代器。
```python
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
        self.add_module("conv1", nn.Conv2d(20 ,10, 4))
model = Model()

for sub_module in model.children():
    print(sub_module)
```
```
Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
Conv2d(20, 10, kernel_size=(4, 4), stride=(1, 1))
```
#### cpu(device_id=None)
将所有的模型参数(`parameters`)和`buffers`复制到`CPU`
`NOTE`：官方文档用的move，但我觉着`copy`更合理。
#### cuda(device_id=None)
将所有的模型参数(`parameters`)和`buffers`赋值`GPU`

参数说明:

- device_id (int, optional) – 如果指定的话，所有的模型参数都会复制到指定的设备上。

#### double()
将`parameters`和`buffers`的数据类型转换成`double`。

#### eval()
将模型设置成`evaluation`模式

仅仅当模型中有`Dropout`和`BatchNorm`是才会有影响。

#### float()
将`parameters`和`buffers`的数据类型转换成`float`。

#### forward(* input)
定义了每次执行的 计算步骤。
在所有的子类中都需要重写这个函数。
#### half()
将`parameters`和`buffers`的数据类型转换成`half`。

#### load_state_dict(state_dict)
将`state_dict`中的`parameters`和`buffers`复制到此`module`和它的后代中。`state_dict`中的`key`必须和 `model.state_dict()`返回的`key`一致。
`NOTE`：用来加载模型参数。

参数说明:

- state_dict (dict) – 保存`parameters`和`persistent buffers`的字典。

#### modules()
返回一个包含 当前模型 所有模块的迭代器。
```python
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
        self.add_module("conv1", nn.Conv2d(20 ,10, 4))
model = Model()

for module in model.modules():
    print(module)
```
```
Model (
  (conv): Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
  (conv1): Conv2d(20, 10, kernel_size=(4, 4), stride=(1, 1))
)
Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
Conv2d(20, 10, kernel_size=(4, 4), stride=(1, 1))
```
可以看出，`modules()`返回的`iterator`不止包含 子模块。这是和`children()`的不同。

**`NOTE：`**
重复的模块只被返回一次(`children()也是`)。 在下面的例子中, `submodule` 只会被返回一次：

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        submodule = nn.Conv2d(10, 20, 4)
        self.add_module("conv", submodule)
        self.add_module("conv1", submodule)
model = Model()

for module in model.modules():
    print(module)
```
```
Model (
  (conv): Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
  (conv1): Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
)
Conv2d(10, 20, kernel_size=(4, 4), stride=(1, 1))
```
#### named_children()
返回 包含 模型当前子模块 的迭代器，`yield` 模块名字和模块本身。

Example

>>> for name, module in model.named_children():
>>>     if name in ['conv4', 'conv5']:
>>>         print(module)
named_modules(memo=None, prefix='')[source]
Returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Note

Duplicate modules are returned only once. In the following example, l will be returned only once.

>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
>>>     print(idx, '->', m)
0 -> ('', Sequential (
  (0): Linear (2 -> 2)
  (1): Linear (2 -> 2)
))
1 -> ('0', Linear (2 -> 2))
parameters(memo=None)[source]
Returns an iterator over module parameters.

This is typically passed to an optimizer.

Example

>>> for param in model.parameters():
>>>     print(type(param.data), param.size())
<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
register_backward_hook(hook)[source]
Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module inputs are computed. The hook should have the following signature:

hook(module, grad_input, grad_output) -> Tensor or None
The grad_input and grad_output may be tuples if the module has multiple inputs or outputs. The hook should not modify its arguments, but it can optionally return a new gradient with respect to input that will be used in place of grad_input in subsequent computations.

This function returns a handle with a method handle.remove() that removes the hook from the module.

register_buffer(name, tensor)[source]
Adds a persistent buffer to the module.

This is typically used to register a buffer that should not to be considered a model parameter. For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

Buffers can be accessed as attributes using given names.

Example

>>> self.register_buffer('running_mean', torch.zeros(num_features))
register_forward_hook(hook)[source]
Registers a forward hook on the module.

The hook will be called every time forward() computes an output. It should have the following signature:

hook(module, input, output) -> None
The hook should not modify the input or output. This function returns a handle with a method handle.remove() that removes the hook from the module.

register_parameter(name, param)[source]
Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.

state_dict(destination=None, prefix='')[source]
Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are included. Keys are corresponding parameter and buffer names.

Example

>>> module.state_dict().keys()
['bias', 'weight']
train(mode=True)[source]
Sets the module in training mode.

This has any effect only on modules such as Dropout or BatchNorm.

zero_grad()[source]
Sets gradients of all model parameters to zero.
