# pytorch 的 hook 机制

在看`pytorch`官方文档的时候，发现在`nn.Module`部分和`Variable`部分均有`hook`的身影。感到很神奇，因为在使用`tensorflow`的时候没有碰到过这个词。所以打算一探究竟。

## nn.Module的hook

### register_forward_hook(hook)
在`module`上注册一个`forward hook`。
每次调用`forward()`计算输出的时候，这个`hook`就会被调用。它应该拥有以下签名：

`hook(module, input, output) -> None`

`hook`不应该修改 `input`和`output`的值。 这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

看这个解释可能有点蒙逼，但是如果要看一下`nn.Module`的源码怎么使用`hook`的话，那就乌云尽散了。
先看 `register_forward_hook`
```python
def register_forward_hook(self, hook):

       handle = hooks.RemovableHandle(self._forward_hooks)
       self._forward_hooks[handle.id] = hook
       return handle
```
这个方法的作用是在此`module`上注册一个`hook`，函数中第一句就没必要在意了，主要看第二句，是把注册的`hook`保存在`_forward_hooks`字典里。

再看 `nn.Module` 的`__call__`方法（被阉割了，只留下需要关注的部分）：

```python

def __call__(self, *input, **kwargs):
   result = self.forward(*input, **kwargs)
   for hook in self._forward_hooks.values():
       #将注册的hook拿出来用
       hook_result = hook(self, input, result)
   ...
   return result
```
可以看到，当我们执行`model(x)`的时候，底层干了以下几件事：

- 调用 `forward` 方法计算结果

- 判断有没有注册 `forward_hook`，有的话，就将 `forward` 的输入及结果作为`hook`的实参。然后让`hook`自己干一些不可告人的事情。

看到这，我们就明白`hook`签名的意思了，还有为什么`hook`不能修改`input`的`output`的原因。

小例子：
```python
import torch
from torch import nn
import torch.functional as F
from torch.autograd import Variable

def for_hook(module, input, output):
    print(module)
    for val in input:
        print("input val:",val)
    for out_val in output:
        print("output val:", out_val)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):

        return x+1

model = Model()
x = Variable(torch.FloatTensor([1]), requires_grad=True)
handle = model.register_forward_hook(for_hook)
print(model(x))
handle.remove()
```


### register_backward_hook

在`module`上注册一个`bachward hook`。

每次计算`module`的`inputs`的梯度的时候，这个`hook`会被调用。`hook`应该拥有下面的`signature`。

`hook(module, grad_input, grad_output) -> Tensor or None`

如果`module`有多个输入输出的话，那么`grad_input` `grad_output`将会是个`tuple`。
`hook`不应该修改它的`arguments`，但是它可以选择性的返回关于输入的梯度，这个返回的梯度在后续的计算中会替代`grad_input`。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

从上边描述来看，`backward hook`似乎可以帮助我们处理一下计算完的梯度。看下面`nn.Module`中`register_backward_hook`方法的实现，和`register_forward_hook`方法的实现几乎一样，都是用字典把注册的`hook`保存起来。
```python
def register_backward_hook(self, hook):
    handle = hooks.RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle
```
先看个例子来看一下`hook`的参数代表了什么：
```python
import torch
from torch import nn
import torch.functional as F
from torch.autograd import Variable

def back_hook(module, grad_in, grad_out):
    print("hello")
    print(grad_in)
    print(grad_out)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        return torch.mean(x+1)

model = Model()
x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=True)
#handle = model.register_forward_hook(for_hook)
model.register_backward_hook(back_hook)
res = model(x)
res.backward()
```
```
hello
(Variable containing:
 0.3333
 0.3333
 0.3333
[torch.FloatTensor of size 3]
,)
(Variable containing:
 1
[torch.FloatTensor of size 1]
,)
```
可以看出，`grad_in`保存的是，此模块`forward`方法的输入的值的梯度。`grad_out`保存的是，此模块`forward`方法返回值的梯度。我们不能在`grad_in`上直接修改，但是我们可以返回一个新的`new_grad_in`作为`forward`方法`input`的梯度。

## Variable 的 hook
### register_hook(hook)
注册一个`backward`钩子。

每次`gradients`被计算的时候，这个`hook`都被调用。`hook`应该拥有以下签名：

`hook(grad) -> Variable or None`

`hook`不应该修改它的输入，但是它可以返回一个替代当前梯度的新梯度。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

例子：
```python
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)  # double the gradient
v.backward(torch.Tensor([1, 1, 1]))
#先计算原始梯度，再进hook，获得一个新梯度。
print(v.grad.data)
h.remove()  # removes the hook
```

```

 2
 2
 2
[torch.FloatTensor of size 3]
```
看到这，我感觉`nn.Module`的`back_hook`功能是借助`Variable`的`register_hook`实现的。

## 其他
最后，猜测一下`pytorch` `backward`的流程。
```python
res.backward()
# 1. res的默认梯度为1
# 2. 将res的梯度 输入到注册在 res 上的 hook，将经hook处理后的梯度作为res的真实梯度。
# 3. 找到res的creator， 将处理后的梯度作为 creator 中 backward()方法的输入，计算backward()实参的梯度，并返回。
# 4. 把backward返回关于input的梯度 输入到注册在input上的hook，经过hook处理返回处理后的梯度作为input的梯度
# 5. input的梯度属性值设置为处理后的 梯度值
# 6. 找到 input 的 creator， 将处理后的梯度作为 creator 中 backward()方法的输入，计算backward() input的梯度，并返回。
# 7. 第4步
```
