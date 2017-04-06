# torch.optim
`torch.optim`是一个实现不同优化算法的包。大多数常用的优化算法都在此包中得以实现，接口设计的也很通用，保证了更复杂的算法也会很容易的集成在这个包里。

如何使用优化器(`optimizer`)：
如果你想使用`torch.optim`的话，首先需要创建一个`optimizer`对象，它会保存当前的状态，并且根据计算出的参数的梯度更新参数。

## **如何构建`optimizer`：**

- 首先，需要给他传递一个 包含要优化的参数的可迭代对象（所有的参数都应该是`Variable`）。

- 之后，你需要指定 `optimizer-specific`选项，例如：学习率、权值衰减(`weight decay`)等等。

例子：

```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)
```

## Per-parameter options(单一参数选项)
优化器同样支持指定每个参数的选项。为了使用这个功能，在传入参数的时候，应该传入一个`dict`列表。列表中的每个`dict`都会定义一个独立的 `参数组`，而且应该包含 `params`键，`params`键对应的值是 一个参数列表。其他的键应该和`optimizer`可接收的键对应，对应的值将会用与这组参数的优化。

**`NOTE：`**
你仍然可以将 `options` 当作 关键字参数传入。如果组`options`没有重写它们的话，它们会当作默认值使用。当你只想改变一个组`option`，其它组保持不变的话，这个特性是非常有用的。

`例子：`
如果想要为每层指定学习率的话，这个方法是非常有用的。
```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```
这串代码意味着，`model.base.parameters()` 将会使用默认的学习率`1e-2`，`model.classifier.parameters()`将会使用学习率`1e-3`，同时这两组参数都会使用 `0.9`的动量。

## 做一步优化
所有的优化器(`optimizers`)都实现了一个`step()`方法，这个方法用来更新注册在这个优化器参数。有两种方法可以使用它：

### `optimizer.step()`
这是一个简化的版本，所有的优化器都支持这个方法。当调用`backward`，梯度被计算完之后，这个方法可以被调用。

例子：

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```
### `optimizer.step(closure)`
一些优化算法，例如`Conjugate Gradient`和`LBFGS`需要多次重新计算。所以，你需要传递一个闭包，优化器可以使用闭包重新计算你的模型。闭包应该将参数的梯度清零，计算loss，然后返回loss。

例子:

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

## 算法
### class torch.optim.Optimizer(params, defaults)[source]
所有`optimizer`的基类。

参数说明：

- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。
- defaults – (dict): 一个`dict`，包含优化选项的默认值。
#### load_state_dict(state_dict)[source]

加载`optimizer`的状态。
参数说明：

- state_dict (dict) – 优化器的状态。应该是由`state_dict()`方法返回的对象。

#### state_dict()[source]

将 优化器的状态作为一个`dict`返回。

它包含两个单元：

- state - 包含当前优化器状态的`state`。不同的优化器，`state`不同。

- param_groups - 包含所有 参数组的 `dict`。

#### step(closure)[source]

执行一次优化动作（参数更新）。

参数说明：

- closure (callable) – 可以重新评估模型和返回`loss` 的闭包。对于多数优化器来说，这个选项是可选择的。

#### zero_grad()[source]

将所要优化的变量的梯度清零。

### class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)[source]

实现了 `adadelta` 算法。

[论文地址](https://arxiv.org/abs/1212.5701)

参数说明：

- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。

- rho (float, optional) – 计算 平方梯度`running average`时所需要的系数。（默认：0.9）

- eps (float, optional) – 加在分母上的值，用于提高数值计算的稳定性（防止除0错误）。（默认值：`1e-6 `）
- lr (float, optional) – 学习率。（默认值：1.0）

- weight_decay (float, optional) – `weight_decay` 系数 (L2 惩罚项) (默认值： 0)

#### step(closure=None)[source]
执行一次优化动作（参数更新）。

参数说明：

- closure (callable， optional) – 可以重新评估模型和返回`loss` 的闭包。

### class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)[source]

实现了`Adagrad`算法。

[论文地址](http://jmlr.org/papers/v12/duchi11a.html)

参数说明：

- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。

- lr (float, optional) – 学习率 (默认值: 1e-2)

- lr_decay (float, optional) – 学习率衰减(默认值： 0)

- weight_decay (float, optional) – `weight_decay` 系数 (L2 penalty) (默认值: 0)
#### step(closure=None)[source]
执行一次优化动作（参数更新）。

参数说明：

- closure (callable，optional) – 可以重新评估模型和返回`loss` 的闭包。

### class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]

实现了`Adam`算法。

[论文地址](https://arxiv.org/abs/1412.6980)

参数说明：
- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。

- lr (float, optional) – 学习率 (默认值： 1e-3)

- betas (Tuple[float, float], optional) – 计算梯度和梯度平方的`running average`时所需的系数。（默认：(0.9, 0.99)）

- eps (float, optional) – 加在分母上的值，用于提高数值计算的稳定性（防止除0错误）。（默认值：`1e-8`）

- weight_decay (float, optional) – `weight_decay`系数 (L2 penalty) (默认值: 0)

#### step(closure=None)[source]

执行一次优化动作（参数更新）。

参数说明：

- closure (callable，optional) – 可以重新评估模型和返回`loss` 的闭包。

### class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]

实现了`Adamax`算法。(Adam的一个变种，在 `infinity norm`情况下).

[论文地址](https://arxiv.org/abs/1412.6980)

参数说明：

- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。

- lr (float, optional) – 学习率 (默认值： `2e-3`)

- betas (Tuple[float, float], optional) – 计算梯度和梯度平方的`running average`时所需的系数。（默认：(0.9, 0.99)）

- eps (float, optional) – 加在分母上的值，用于提高数值计算的稳定性（防止除0错误）。（默认值：`1e-8`）

- weight_decay (float, optional) – `weight_decay`系数 (L2 penalty) (默认值: 0)

#### step(closure=None)[source]

执行一次优化动作（参数更新）。

参数说明：

- closure (callable，optional) – 可以重新评估模型和返回`loss` 的闭包。

### class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)[source]

实现了`ASGD`算法。`ASGD`(`Averaged Stochastic Gradient Descent`)。
[论文地址](http://dl.acm.org/citation.cfm?id=131098)

参数说明：

- params (iterable) – 包含变量或者`dict`的可迭代对象。指定有哪些变量需要被优化。

- lr (float, optional) – 学习率 (默认值： `1e-2`)

- lambd (float, optional) – 衰减项（`decay term`） (默认值： `1e-4`)

- alpha (float, optional) – power for eta update (default: 0.75)

- t0 (float, optional) – point at which to start averaging (default: 1e6)

- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

#### step(closure=None)[source]
执行一次优化动作（参数更新）。

参数说明：

- closure (callable，optional) – 可以重新评估模型和返回`loss` 的闭包。

### class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)[source]
Implements L-BFGS algorithm.

Warning

This optimizer doesn’t support per-parameter options and parameter groups (there can be only one).

Warning

Right now all parameters have to be on a single device. This will be improved in the future.

Note

This is a very memory intensive optimizer (it requires additional param_bytes * (history_size + 1) bytes). If it doesn’t fit in memory try reducing the history size, or use a different algorithm.

Parameters:
lr (float) – learning rate (default: 1)
max_iter (int) – maximal number of iterations per optimization step (default: 20)
max_eval (int) – maximal number of function evaluations per optimization step (default: max_iter * 1.25).
tolerance_grad (float) – termination tolerance on first order optimality (default: 1e-5).
tolerance_change (float) – termination tolerance on function value/parameter changes (default: 1e-9).
history_size (int) – update history size (default: 100).
step(closure)[source]
Performs a single optimization step.

Parameters:	closure (callable) – A closure that reevaluates the model and returns the loss.
### class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)[source]
Implements RMSprop algorithm.

Proposed by G. Hinton in his course.

The centered version first appears in Generating Sequences With Recurrent Neural Networks.

Parameters:
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
lr (float, optional) – learning rate (default: 1e-2)
momentum (float, optional) – momentum factor (default: 0)
alpha (float, optional) – smoothing constant (default: 0.99)
eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
centered (bool, optional) – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
step(closure=None)[source]
Performs a single optimization step.

Parameters:	closure (callable, optional) – A closure that reevaluates the model and returns the loss.
### class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))[source]
Implements the resilient backpropagation algorithm.

Parameters:
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
lr (float, optional) – learning rate (default: 1e-2)
etas (Tuple[float, float], optional) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors (default: (0.5, 1.2))
step_sizes (Tuple[float, float], optional) – a pair of minimal and maximal allowed step sizes (default: (1e-6, 50))
step(closure=None)[source]
Performs a single optimization step.

Parameters:	closure (callable, optional) – A closure that reevaluates the model and returns the loss.

### class torch.optim.SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)[source]
Implements stochastic gradient descent (optionally with momentum).

Nesterov momentum is based on the formula from On the importance of initialization and momentum in deep learning.

Parameters:
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
lr (float) – learning rate
momentum (float, optional) – momentum factor (default: 0)
weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
dampening (float, optional) – dampening for momentum (default: 0)
nesterov (bool, optional) – enables Nesterov momentum (default: False)
Example

>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
step(closure=None)[source]
Performs a single optimization step.

Parameters:	closure (callable, optional) – A closure that reevaluates the model and returns the loss.
