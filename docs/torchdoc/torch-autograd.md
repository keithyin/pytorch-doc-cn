# Automatic differentiation package - torch.autograd

`torch.autograd`提供了类和函数用来对任意标量函数进行求导。要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的`tensor`包含进`Variable`对象中即可。

### torch.autograd.backward(variables, grad_variables, retain_variables=False)
Computes the sum of gradients of given variables w.r.t. graph leaves.
给定图的叶子节点`variables`, 计算图中变量的梯度和。
计算图可以通过链式法则求导。如果`variables`中的任何一个`variable`是 非标量(`non-scalar`)的，且需要`requires_grad=True`。那么此函数需要指定`grad_variables`，它的长度应该和`variables`的长度匹配，里面保存了相关`variable`的梯度(对于不需要`gradient tensor`的`variable`，`None`是可取的)。

此函数累积叶子节点计算的梯度。你可能需要在调用此函数之前将`variable`的梯度置零。

参数说明:

- variables (variable 列表) – 被求微分的叶子节点，即 `ys` 。

- grad_variables (`Tensor` 列表) – 对应`variable`的梯度。仅当`variable`不是标量且需要求梯度的时候使用。

- retain_variables (bool) – `True`,计算梯度时所需要的`buffer`在计算完梯度后不会被释放。如果想对一个子图多次求微分的话，需要设置为`True`。

## Variable
### API 兼容性

`Variable API` 几乎和 `Tensor API`一致 (除了一些`in-place`方法，这些`in-place`方法会修改 `required_grad=True`的 `input` 的值)。多数情况下，将`Tensor`替换为`Variable`，代码一样会正常的工作。由于这个原因，我们不会列出`Variable`的所有方法，你可以通过`torch.Tensor`的文档来获取相关知识。

### In-place operations on Variables
在`autograd`中支持`in-place operations`是非常困难的。同时在很多情况下，我们阻止使用`in-place operations`。`Autograd`的贪婪的 释放`buffer`和 复用使得它效率非常高。只有在非常少的情况下，使用`in-place operations`可以降低内存的使用。除非你面临很大的内存压力，否则不要使用`in-place operations`。

### In-place 正确性检查
所有的`Variable`都会记录用在他们身上的 `in-place operations`，
All Variable s keep track of in-place operations applied to them, and if the implementation detects that a variable was saved for backward in one of the functions, but it was modified in-place afterwards, an error will be raised once backward pass is started. This ensures that if you’re using in-place functions and not seing any errors, you can be sure that the computed gradients are correct.
