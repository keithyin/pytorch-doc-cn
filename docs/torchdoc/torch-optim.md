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
一些优化算法，例如`Conjugate Gradient`和`LBFGS`需要多次重新计算函数。
Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.

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
