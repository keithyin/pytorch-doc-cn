# Automatic differentiation package - torch.autograd

`torch.autograd`提供了类和函数用来对任意标量函数进行求导。要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的`tensor`包含进`Variable`对象中即可。

### torch.autograd.backward(variables, grad_variables, retain_variables=False)
Computes the sum of gradients of given variables w.r.t. graph leaves.

The graph is differentiated using the chain rule. If any of variables are non-scalar (i.e. their data has more than one element) and require gradient, the function additionaly requires specifying grad_variables. It should be a sequence of matching length, that containins gradient of the differentiated function w.r.t. corresponding variables (None is an acceptable value for all variables that don’t need gradient tensors).

This function accumulates gradients in the leaves - you might need to zero them before calling it.

Parameters:
variables (sequence of Variable) – Variables of which the derivative will be computed.
grad_variables (sequence of Tensor) – Gradients w.r.t. each element of corresponding variables. Required only for non-scalar variables that require gradient.
retain_variables (bool) – If True, buffers necessary for computing gradients won’t be freed after use. It is only necessary to specify True if you want to differentiate some subgraph multiple times.
