# PyTorch结构介绍
对PyTorch架构的粗浅理解，不能保证完全正确，但是希望可以从更高层次上对PyTorch上有个整体把握。水平有限，如有错误，欢迎指错，谢谢！

## 几个重要的类型
### 和数值相关的
* Tensor
* Variable
* Parameter
* buffer(这个其实不能叫做类型，其实他就是用来保存tensor的)

**Tensor**:
`PyTorch`中的计算基本都是基于`Tensor`的，可以说是`PyTorch`中的基本计算单元。

**Variable**：
`Tensor`的一个`Wrapper`，其中保存了`Variable`的创造者，`Variable`的值（tensor），还有`Variable`的梯度(`Variable`)。

自动求导机制的核心组件，因为它不仅保存了 变量的值，还保存了变量是由哪个`op`产生的。这在反向传导的过程中是十分重要的。

`Variable`的前向过程的计算包括两个部分的计算，一个是其值的计算（即，Tensor的计算），还有就是`Variable`标签的计算。标签指的是什么呢？如果您看过PyTorch的官方文档 `Excluding subgraphs from backward` 部分的话，您就会发现`Variable`还有两个标签：`requires_grad`和`volatile`。标签的计算指的就是这个。

**Parameter**:
这个类是`Variable`的一个子集，`PyTorch`给出这个类的定义是为了在`Module`(下面会谈到)中添加模型参数方便。

### 模型相关的
* Function
* Module

**Function**:
如果您想在`PyTorch`中自定义`OP`的话，您需要继承这个类，您需要在继承的时候复写`forward`和`backward`方法，可能还需要复写`__init__`方法（由于篇幅控制，这里不再详细赘述如果自定义`OP`）。您需要在`forward`中定义`OP`，在`backward`说明如何计算梯度。
*关于`Function`，还需要知道的一点就是，`Function`中`forward`和`backward`方法中进行计算的类型都是`Tensor`，而不是我们传入的Variable。计算完forward和backward之后，会包装成Varaible返回。这种设定倒是可以理解的，因为OP是一个整体嘛，OP内部的计算不需要记录creator*

**Module**:
这个类和`Function`是有点区别的，回忆一下，我们定义`Function`的时候，`Funciton`本身是不需要变量的，而`Module`是变量和`Function`的结合体。在某些时候，我们更倾向称这种结构为`Layer`。但是这里既然这么叫，那就这么叫吧。

`Module`实际上是一个容器，我们可以继承`Module`，在里面加几个参数，从而实现一个简单全连接层。我们也可以继承`Module`，在里面加入其它`Module`，从而实现整个`VGG`结构。

## 关于hook
**PyTorch中注册的hook都是不允许改变hook的输入值的**
下面对PyTorch中出现hook的地方做个总结：
* Module : register_forward_hook, register_backward_hook
注意：forward_hook不能用来修改Module的输出值，它的功能就像是安装个监视器一样。我们可以用forward_hook和visdom来监控我们Module的输出。backward_hook和与`Variable`的功能是类似的，将和`Variable`的`register_hook`一起介绍。

* Variable: register_hook
Variable的register_hook注册的是一个`backward hook`，`backward hook`是在BP的过程中会用到的。可以用它来处理计算的梯度。

## foward过程与backward过程
**forward**
以一个Module为例：
1. 调用module的`call`方法
2. `module`的`call`里面调用`module`的`forward`方法
3. `forward`里面如果碰到`Module`的子类，回到第1步，如果碰到的是`Function`的子类，继续往下
4. 调用`Function`的`call`方法
5. `Function`的`call`方法调用了Function的`forward`方法。
6. `Function`的`forward`返回值
7. `module`的`forward`返回值
8. 在`module`的`call`进行`forward_hook`操作，然后返回值。

**backward**
```python
res.backward()
# 1. res的默认梯度为1
# 2. 将res的梯度 输入到注册在 res 上的 hook，将经hook处理后的梯度作为res的真实梯度。
# 3. 找到res的creator， 将处理后的梯度作为 creator 中 backward()方法的输入，计算backward()实参的梯度，并返回。
# 4. 把backward返回关于input的梯度 输入到注册在input上的hook，经过hook处理返回处理后的梯度作为input的梯度
# 5. input的梯度属性值设置为处理后的 梯度值
# 6. 找到 input 的 creator， 将处理后的梯度作为 creator 中 backward()方法的输入，计算backward() input的梯度，并返回。
# 7. 第4步
# 如果恰巧碰到了对于一个Varaible，既有Module的backward_hook也有自己注册的backward_hook，那么执行的顺序是，
# 先module的，后自己的。
```

## 总结
PyTorch基本的操作是`OP`，被操作数是`Tensor`。
