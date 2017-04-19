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
        #self.conv = nn.Conv2d(10, 20, 4) 和上面这个增加module的方式等价
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

例子：
```python
for name, module in model.named_children():
    if name in ['conv4', 'conv5']:
        print(module)
```
#### named_modules(memo=None, prefix='')[source]

返回包含网络中所有模块的迭代器, `yielding`  模块名和模块本身。

**`注意：`**

重复的模块只被返回一次(`children()也是`)。 在下面的例子中, `submodule` 只会被返回一次。


#### parameters(memo=None)

返回一个 包含模型所有参数 的迭代器。

一般用来当作`optimizer`的参数。

例子：
```python
for param in model.parameters():
    print(type(param.data), param.size())

<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
```
#### register_backward_hook(hook)

在`module`上注册一个`bachward hook`。

每次计算`module`的`inputs`的梯度的时候，这个`hook`会被调用。`hook`应该拥有下面的`signature`。

`hook(module, grad_input, grad_output) -> Variable or None`

如果`module`有多个输入输出的话，那么`grad_input` `grad_output`将会是个`tuple`。
`hook`不应该修改它的`arguments`，但是它可以选择性的返回关于输入的梯度，这个返回的梯度在后续的计算中会替代`grad_input`。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。


#### register_buffer(name, tensor)

给`module`添加一个`persistent buffer`。

`persistent buffer`通常被用在这么一种情况：我们需要保存一个状态，但是这个状态不能看作成为模型参数。
例如：, `BatchNorm’s` running_mean 不是一个 `parameter`, 但是它也是需要保存的状态之一。

`Buffers`可以通过注册时候的`name`获取。

**`NOTE`:我们可以用 buffer 保存 `moving average`**

例子：

```python
self.register_buffer('running_mean', torch.zeros(num_features))

self.running_mean
```

#### register_forward_hook(hook)

在`module`上注册一个`forward hook`。
每次调用`forward()`计算输出的时候，这个`hook`就会被调用。它应该拥有以下签名：

`hook(module, input, output) -> None`

`hook`不应该修改 `input`和`output`的值。 这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。


#### register_parameter(name, param)
向`module`添加 `parameter`

`parameter`可以通过注册时候的`name`获取。

#### state_dict(destination=None, prefix='')[source]

返回一个字典，保存着`module`的所有状态（`state`）。

`parameters`和`persistent buffers`都会包含在字典中，字典的`key`就是`parameter`和`buffer`的 `names`。

例子：
```python
import torch
from torch.autograd import Variable
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2 = nn.Linear(1, 2)
        self.vari = Variable(torch.rand([1]))
        self.par = nn.Parameter(torch.rand([1]))
        self.register_buffer("buffer", torch.randn([2,3]))

model = Model()
print(model.state_dict().keys())

```
```
odict_keys(['par', 'buffer', 'conv2.weight', 'conv2.bias'])
```
#### train(mode=True)

将`module`设置为 `training mode`。

仅仅当模型中有`Dropout`和`BatchNorm`是才会有影响。

#### zero_grad()

将`module`中的所有模型参数的梯度设置为0.

### class torch.nn.Sequential(* args)

一个时序容器。`Modules` 会以他们传入的顺序被添加到容器中。当然，也可以传入一个`OrderedDict`。

为了更容易的理解如何使用`Sequential`, 下面给出了一个例子:

```python
# Example of using Sequential

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

### class torch.nn.ModuleList(modules=None)[source]
将`submodules`保存在一个`list`中。

`ModuleList`可以像一般的`Python list`一样被`索引`。而且`ModuleList`中包含的`modules`已经被正确的注册，对所有的`module method`可见。


参数说明:

- modules (list, optional) – 将要被添加到`MuduleList`中的 `modules` 列表

例子:
```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

####  append(module)[source]
等价于 list 的 `append()`

参数说明:

- module (nn.Module) – 要 append 的`module`
#### extend(modules)[source]
等价于 `list` 的 `extend()` 方法

参数说明:

- modules (list) – list of modules to append

### class torch.nn.ParameterList(parameters=None)
将`submodules`保存在一个`list`中。

`ParameterList`可以像一般的`Python list`一样被`索引`。而且`ParameterList`中包含的`parameters`已经被正确的注册，对所有的`module method`可见。


参数说明:

- modules (list, optional) – a list of nn.Parameter

例子:
```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x
```
#### append(parameter)[source]
等价于` python list` 的 `append` 方法。

参数说明:

- parameter (nn.Parameter) – parameter to append
#### extend(parameters)[source]
等价于` python list` 的 `extend` 方法。

参数说明:

- parameters (list) – list of parameters to append

## Convolution Layers

## Pooling Layers

## Non-linear Activations

## Normalization layers

## Recurrent layers
### class torch.nn.RNN(* args, ** kwargs)[source]

将一个多层的 `Elman RNN`，激活函数为`tanh`或者`ReLU`，用于输入序列。

对输入序列中每个元素，`RNN`每层的计算公式为
$$
h_t=tanh(w_{ih}* x_t+b_{ih}+w_{hh}* h_{t-1}+b_{hh})
$$
$h_t$是时刻$t$的隐状态。 $x_t$是上一层时刻$t$的隐状态，或者是第一层在时刻$t$的输入。如果`nonlinearity='relu'`,那么将使用`relu`代替`tanh`作为激活函数。

参数说明:

- input_size – 输入`x`的特征数量。

- hidden_size – 隐层的特征数量。

- num_layers – RNN的层数。

- nonlinearity – 指定非线性函数使用`tanh`还是`relu`。默认是`tanh`。

- bias – 如果是`False`，那么RNN层就不会使用偏置权重 $b_ih$和$b_hh$,默认是`True`

- batch_first – 如果`True`的话，那么输入`Tensor`的shape应该是[batch_size, time_step, feature],输出也是这样。
- dropout – 如果值非零，那么除了最后一层外，其它层的输出都会套上一个`dropout`层。

- bidirectional – 如果`True`，将会变成一个双向`RNN`，默认为`False`。


`RNN`的输入：
**(input, h_0)**
- input (seq_len, batch, input_size): 保存输入序列特征的`tensor`。`input`可以是被填充的变长的序列。细节请看`torch.nn.utils.rnn.pack_padded_sequence()`

- h_0 (num_layers * num_directions, batch, hidden_size): 保存着初始隐状态的`tensor`

`RNN`的输出：
**(output, h_n)**

- output (seq_len, batch, hidden_size * num_directions): 保存着`RNN`最后一层的输出特征。如果输入是被填充过的序列，那么输出也是被填充的序列。
- h_n (num_layers * num_directions, batch, hidden_size): 保存着最后一个时刻隐状态。

`RNN`模型参数:

- weight_ih_l[k] – 第`k`层的 `input-hidden` 权重， 可学习，形状是`(input_size x hidden_size)`。

- weight_hh_l[k] – 第`k`层的 `hidden-hidden` 权重， 可学习，形状是`(hidden_size x hidden_size)`

- bias_ih_l[k] – 第`k`层的 `input-hidden` 偏置， 可学习，形状是`(hidden_size)`

- bias_hh_l[k] – 第`k`层的 `hidden-hidden` 偏置， 可学习，形状是`(hidden_size)`

示例：
```python
rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```
### class torch.nn.LSTM(* args, ** kwargs)[source]

将一个多层的 `(LSTM)` 应用到输入序列。

对输入序列的每个元素，`LSTM`的每层都会执行以下计算：
$$
\begin{aligned}
i_t &= sigmoid(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}) \\
f_t &= sigmoid(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}) \\
o_t &= sigmoid(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})\\
g_t &= tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})\\
c_t &= f_t*c_{t-1}+i_t*g_t\\
h_t &= o_t*tanh(c_t)
\end{aligned}
$$
$h_t$是时刻$t$的隐状态,$c_t$是时刻$t$的细胞状态，$x_t$是上一层的在时刻$t$的隐状态或者是第一层在时刻$t$的输入。$i_t, f_t, g_t, o_t$ 分别代表 输入门，遗忘门，细胞和输出门。

参数说明:

- input_size – 输入的特征维度

- hidden_size – 隐状态的特征维度

- num_layers – 层数（和时序展开要区分开）

- bias – 如果为`False`，那么`LSTM`将不会使用$b_{ih},b_{hh}$，默认为`True`。

- batch_first – 如果为`True`，那么输入和输出`Tensor`的形状为`(batch, seq, feature)`

- dropout – 如果非零的话，将会在`RNN`的输出上加个`dropout`，最后一层除外。

- bidirectional – 如果为`True`，将会变成一个双向`RNN`，默认为`False`。

`LSTM`输入:
input, (h_0, c_0)

- input (seq_len, batch, input_size): 包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])

- h_0 (num_layers * num_directions, batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

- c_0 (num_layers * num_directions, batch, hidden_size): 保存着`batch`中每个元素的初始化细胞状态的`Tensor`

`LSTM`输出
output, (h_n, c_n)

- output (seq_len, batch, hidden_size * num_directions): 保存`RNN`最后一层的输出的`Tensor`。 如果输入是`torch.nn.utils.rnn.PackedSequence`，那么输出也是`torch.nn.utils.rnn.PackedSequence`。

- h_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的隐状态。

- c_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的细胞状态。

`LSTM`模型参数:

- weight_ih_l[k] – 第`k`层可学习的`input-hidden`权重($W_{ii}|W_{if}|W_{ig}|W_{io}$)，形状为`(input_size x 4*hidden_size)`

- weight_hh_l[k] – 第`k`层可学习的`hidden-hidden`权重($W_{hi}|W_{hf}|W_{hg}|W_{ho}$)，形状为`(hidden_size x 4*hidden_size)`。

- bias_ih_l[k] – 第`k`层可学习的`input-hidden`偏置($b_{ii}|b_{if}|b_{ig}|b_{io}$)，形状为`( 4*hidden_size)`

- bias_hh_l[k] – 第`k`层可学习的`hidden-hidden`偏置($b_{hi}|b_{hf}|b_{hg}|b_{ho}$)，形状为`( 4*hidden_size)`。
示例:
```python
lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))
```

### class torch.nn.GRU(* args, ** kwargs)[source]

将一个多层的`GRU`用于输入序列。

对输入序列中的每个元素，每层进行了一下计算：

$$
\begin{aligned}
r_t&=sigmoid(W_{ir}x_t+b_{ir}+W_{hr}h_{(t-1)}+b_{hr})\\
i_t&=sigmoid(W_{ii}x_t+b_{ii}+W_{hi}h_{(t-1)}+b_{hi})\\
n_t&=tanh(W_{in}x_t+b_{in}+rt*(W_{hn}h_{(t-1)}+b_{hn}))\\
h_t&=(1-i_t)* nt+i_t*h(t-1)
\end{aligned}
$$
$h_t$是是时间$t$的上的隐状态，$x_t$是前一层$t$时刻的隐状态或者是第一层的$t$时刻的输入，$r_t, i_t, n_t$分别是重置门，输入门和新门。

参数说明：
- input_size – 期望的输入$x$的特征值的维度
- hidden_size – 隐状态的维度
- num_layers – `RNN`的层数。
- bias – 如果为`False`，那么`RNN`层将不会使用`bias`，默认为`True`。
- batch_first – 如果为`True`的话，那么输入和输出的`tensor`的形状是`(batch, seq, feature)`。
- dropout –  如果非零的话，将会在`RNN`的输出上加个`dropout`，最后一层除外。
- bidirectional – 如果为`True`，将会变成一个双向`RNN`，默认为`False`。

输入：
input, h_0

- input (seq_len, batch, input_size):  包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])。

- h_0 (num_layers * num_directions, batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

输出：
output, h_n

- output (seq_len, batch, hidden_size * num_directions): ten保存`RNN`最后一层的输出的`Tensor`。 如果输入是`torch.nn.utils.rnn.PackedSequence`，那么输出也是`torch.nn.utils.rnn.PackedSequence`。

- h_n (num_layers * num_directions, batch, hidden_size): `Tensor`，保存着`RNN`最后一个时间步的隐状态。

变量：

- weight_ih_l[k] – 第`k`层可学习的`input-hidden`权重($W_{ir}|W_{ii}|W_{in}$)，形状为`(input_size x 3*hidden_size)`

- weight_hh_l[k] – 第`k`层可学习的`hidden-hidden`权重($W_{hr}|W_{hi}|W_{hn}$)，形状为`(hidden_size x 3*hidden_size)`。

- bias_ih_l[k] – 第`k`层可学习的`input-hidden`偏置($b_{ir}|b_{ii}|b_{in}$)，形状为`( 3*hidden_size)`

- bias_hh_l[k] – 第`k`层可学习的`hidden-hidden`偏置($b_{hr}|b_{hi}|b_{hn}$)，形状为`( 3*hidden_size)`。


例子：
```python
 rnn = nn.GRU(10, 20, 2)
 input = Variable(torch.randn(5, 3, 10))
 h0 = Variable(torch.randn(2, 3, 20))
 output, hn = rnn(input, h0)
```

### class torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')[source]

一个 `Elan RNN cell`，激活函数是`tanh`或`ReLU`，用于输入序列。
将一个多层的 `Elman RNNCell`，激活函数为`tanh`或者`ReLU`，用于输入序列。
$$
h'=tanh(w_{ih}* x+b_{ih}+w_{hh}* h+b_{hh})
$$
如果`nonlinearity=relu`，那么将会使用`ReLU`来代替`tanh`。

参数：

- input_size – 输入$x$，特征的维度。

- hidden_size – 隐状态特征的维度。

- bias – 如果为`False`，`RNN cell`中将不会加入`bias`，默认为`True`。

- nonlinearity – 用于选择非线性激活函数 [`tanh`|`relu`]. 默认值为： `tanh`

输入：
input, hidden

- input (batch, input_size): 包含输入特征的`tensor`。

- hidden (batch, hidden_size): 保存着初始隐状态值的`tensor`。

输出： h’

- h’ (batch, hidden_size):下一个时刻的隐状态。

变量：

- weight_ih –  `input-hidden` 权重， 可学习，形状是`(input_size x hidden_size)`。

- weight_hh –  `hidden-hidden` 权重， 可学习，形状是`(hidden_size x hidden_size)`

- bias_ih –  `input-hidden` 偏置， 可学习，形状是`(hidden_size)`

- bias_hh –  `hidden-hidden` 偏置， 可学习，形状是`(hidden_size)`

例子：

```python
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx = rnn(input[i], hx)
   output.append(hx)
```

### class torch.nn.LSTMCell(input_size, hidden_size, bias=True)[source]

`LSTM cell`。
$$
\begin{aligned}
i &= sigmoid(W_{ii}x+b_{ii}+W_{hi}h+b_{hi}) \\
f &= sigmoid(W_{if}x+b_{if}+W_{hf}h+b_{hf}) \\
o &= sigmoid(W_{io}x+b_{io}+W_{ho}h+b_{ho})\\
g &= tanh(W_{ig}x+b_{ig}+W_{hg}h+b_{hg})\\
c' &= f_t*c_{t-1}+i_t*g_t\\
h' &= o_t*tanh(c')
\end{aligned}
$$

参数：

- input_size – 输入的特征维度。
- hidden_size – 隐状态的维度。
- bias – 如果为`False`，那么将不会使用`bias`。默认为`True`。

`LSTM`输入:
input, (h_0, c_0)

- input (seq_len, batch, input_size): 包含输入序列特征的`Tensor`。也可以是`packed variable` ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])

- h_0 ( batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

- c_0 (batch, hidden_size): 保存着`batch`中每个元素的初始化细胞状态的`Tensor`

输出：
h_1, c_1

- h_1 (batch, hidden_size): 下一个时刻的隐状态。
- c_1 (batch, hidden_size): 下一个时刻的细胞状态。

`LSTM`模型参数:

- weight_ih – `input-hidden`权重($W_{ii}|W_{if}|W_{ig}|W_{io}$)，形状为`(input_size x 4*hidden_size)`

- weight_hh – `hidden-hidden`权重($W_{hi}|W_{hf}|W_{hg}|W_{ho}$)，形状为`(hidden_size x 4*hidden_size)`。

- bias_ih – `input-hidden`偏置($b_{ii}|b_{if}|b_{ig}|b_{io}$)，形状为`( 4*hidden_size)`

- bias_hh – `hidden-hidden`偏置($b_{hi}|b_{hf}|b_{hg}|b_{ho}$)，形状为`( 4*hidden_size)`。

Examples:
```python
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx, cx = rnn(input[i], (hx, cx))
   output.append(hx)
```

### class torch.nn.GRUCell(input_size, hidden_size, bias=True)[source]

一个`GRU cell`。
$$
\begin{aligned}
r&=sigmoid(W_{ir}x+b_{ir}+W_{hr}h+b_{hr})\\
i&=sigmoid(W_{ii}x+b_{ii}+W_{hi}h+b_{hi})\\
n&=tanh(W_{in}x+b_{in}+r*(W_{hn}h+b_{hn}))\\
h'&=(1-i)* n+i*h
\end{aligned}
$$

参数说明：
- input_size – 期望的输入$x$的特征值的维度
- hidden_size – 隐状态的维度
- bias – 如果为`False`，那么`RNN`层将不会使用`bias`，默认为`True`。

输入：
input, h_0

- input (batch, input_size):  包含输入特征的`Tensor`

- h_0 (batch, hidden_size):保存着`batch`中每个元素的初始化隐状态的`Tensor`

输出：
h_1

- h_1 (batch, hidden_size): `Tensor`，保存着`RNN`下一个时刻的隐状态。

变量：

- weight_ih – `input-hidden`权重($W_{ir}|W_{ii}|W_{in}$)，形状为`(input_size x 3*hidden_size)`

- weight_hh – `hidden-hidden`权重($W_{hr}|W_{hi}|W_{hn}$)，形状为`(hidden_size x 3*hidden_size)`。

- bias_ih – `input-hidden`偏置($b_{ir}|b_{ii}|b_{in}$)，形状为`( 3*hidden_size)`

- bias_hh – `hidden-hidden`偏置($b_{hr}|b_{hi}|b_{hn}$)，形状为`( 3*hidden_size)`。

例子：
```python
rnn = nn.GRUCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx = rnn(input[i], hx)
   output.append(hx)
```
## Linear layers
### class torch.nn.Linear(in_features, out_features, bias=True)[source]
对输入数据进行先行变换。`y = xA + b` （注意：x在前）

参数：

- in_features – 输入样本的特征维度
- out_features – 样本的输出维度
- bias – 如果为`false`，不会加上偏置。默认为`True`

形状：

- Input: `(N, in_features)`

- Output: `(N, out_features)`

变量：

- weight – 模块参数，可学习的线性变换的权重`(out_features x in_features)`
- bias – 模块参数，可学习的线性变换的偏置 `(out_features)`

例子：
```python
>>> m = nn.Linear(20, 30)
>>> input = autograd.Variable(torch.randn(128, 20))
>>> output = m(input)
>>> print(output.size())

```
## Dropout layers
### class torch.nn.Dropout(p=0.5, inplace=False)[source]

将`input tensor`随机置零。

参数说明：
- p – 将`input tensor`随机置零的概率. 默认值为 0.5。

- inplace – 如果设置为`True`将会执行`in-place`操作，默认为`False`。

形状：
- Input: Any. Input 可以是任何形状
- Output: Output的形状与输入一样。

例子：
```python
>>> m = nn.Dropout(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16))
>>> output = m(input)
```

### class torch.nn.Dropout2d(p=0.5, inplace=False)[source]

随机将整个`channel`置零。（将某`channel`上的值置零）

通常情况下，input是Conv2D 模块的输出。

论文`Efficient Object Localization Using Convolutional Networks`中提到过，如果`feature maps`之间相邻的像素是强相关的（卷积层的前几层通常会出现这种情况），那么独立同分布的`dropout`不会正则化激活值，还会导致有效的学习率下降。
>As described in the paper Efficient Object Localization Using Convolutional Networks , if adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then iid dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease.

在这种情况下，使用`nn.Dropout2d()`会提高`feature map`间的独立性。

```python
inputs[i, sampled_c, :, :] = 0
```
参数：

- p (float, optional) – 置零的概率

- inplace (bool, optional) – 如果为`True`，将会执行`in-place`操作。

形状：

- Input: `(N, C, H, W)`

- Output: 和输入同形状。

例子：
```python
>>> m = nn.Dropout2d(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16, 32, 32))
>>> output = m(input)
```
### class torch.nn.Dropout3d(p=0.5, inplace=False)[source]

和上面一样，只不过这个用于3D卷积的输出的。
```python
inputs[i, sampled_c, :, :, :] = 0
```

Parameters:
p (float, optional) – probability of an element to be zeroed.
inplace (bool, optional) – If set to True, will do this operation in-place
Shape:
Input: \((N, C, D, H, W)\)
Output: \((N, C, D, H, W)\) (same shape as input)

例子：
```python
>>> m = nn.Dropout3d(p=0.2)
>>> input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
>>> output = m(input)
```
## Sparse layers
### class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
一个简单的`lookup table`，用来保存固定字典长度和`embedding size`的`embeddings`。

这个模块通常用来保存字向量`work vector`,使用索引来获取它们。这个模块的输入是一个 包含 `indices` 的列表，输出是对应的`word embeddings`。


参数说明：

- num_embeddings (int) – 要进行`embedding`的字典的大小。

- embedding_dim (int) – `embedding`的维度

- padding_idx (int, optional) – 不知道这是干嘛的。If given, pads the output with zeros whenever it encounters the index.

- max_norm (float, optional) – 如果提供的话，`pytorch`将会重新正则化以保证`embeddings`的`norm`小于给定的值。

- norm_type (float, optional) – 指定 `p-norm`的`p`值。

- scale_grad_by_freq (boolean, optional) – 如果提供的话，将会使用字典中词的词频来调整计算的梯度。

变量：

- weight (Tensor) – 模块中可学习的权重，shape为`(num_embeddings, embedding_dim)`。

形状：

- Input: Variable of LongTensor (N, W), N = mini-batch, W = `mini-batch`中每个样本的`indices`个数。

- Output: (N, W, embedding_dim)

总的来说：`emb = nn.Embedding(..,..)`创建了一个`embeddings`矩阵，`emb(indices)`来获取相应的`embedding`。
例子：
```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
>>> embedding(input)

Variable containing:
(0 ,.,.) =
 -1.0822  1.2522  0.2434
  0.8393 -0.6062 -0.3348
  0.6597  0.0350  0.0837
  0.5521  0.9447  0.0498

(1 ,.,.) =
  0.6597  0.0350  0.0837
 -0.1527  0.0877  0.4260
  0.8393 -0.6062 -0.3348
 -0.8738 -0.9054  0.4281
[torch.FloatTensor of size 2x4x3]

>>> # example with padding_idx
>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> input = Variable(torch.LongTensor([[0,2,0,5]]))
>>> embedding(input)

Variable containing:
(0 ,.,.) =
  0.0000  0.0000  0.0000
  0.3452  0.4937 -0.9361
  0.0000  0.0000  0.0000
  0.0706 -2.1962 -0.6276
[torch.FloatTensor of size 1x4x3]
```

## Distance functions


## Loss functions
基本用法：
```python
criterion = LossCriterion() #构造函数有自己的参数
loss = criterion(x, y) #调用标准时也有参数
```
计算出来的结果已经对`mini-batch`取了平均。
### class torch.nn.L1Loss(size_average=True)[source]
创建一个衡量输入`x`(`模型预测输出`)和目标`y`之间差的绝对值的平均值的标准。
$$
loss(x,y)=1/n\sum|x_i-y_i|
$$

- `x` 和 `y` 可以是任意形状，每个包含`n`个元素。

- 对`n`个元素对应的差值的绝对值求和，得出来的结果除以`n`。

- 如果在创建`L1Loss`实例的时候在构造函数中传入`size_average=False`，那么求出来的绝对值的和将不会除以`n`


### class torch.nn.MSELoss(size_average=True)[source]
创建一个衡量输入`x`(`模型预测输出`)和目标`y`之间均方误差标准。
$$
loss(x,y)=1/n\sum(x_i-y_i)^2
$$

- `x` 和 `y` 可以是任意形状，每个包含`n`个元素。

- 对`n`个元素对应的差值的绝对值求和，得出来的结果除以`n`。

- 如果在创建`MSELoss`实例的时候在构造函数中传入`size_average=False`，那么求出来的平方和将不会除以`n`

### class torch.nn.CrossEntropyLoss(weight=None, size_average=True)[source]
此标准将`LogSoftMax`和`NLLLoss`集成到一个类中。

当训练一个多类分类器的时候，这个方法是十分有用的。

- weight(tensor): `1-D` tensor，`n`个元素，分别代表`n`类的权重，如果你的训练样本很不均衡的话，是非常有用的。默认值为None。

调用时参数：

- input : 包含每个类的得分，`2-D` tensor,`shape`为 `batch*n`

- target: 大小为 `n` 的 `1—D` `tensor`，包含类别的索引(`0到 n-1`)。


Loss可以表述为以下形式：
$$
\begin{aligned}
loss(x, class) &= -\text{log}\frac{exp(x[class])}{\sum_j exp(x[j]))}\\
               &= -x[class] + log(\sum_j exp(x[j]))
\end{aligned}
$$
当`weight`参数被指定的时候，`loss`的计算公式变为：
$$
loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
$$
计算出的`loss`对`mini-batch`的大小取了平均。

形状(`shape`)：

- Input: (N,C) `C` 是类别的数量

- Target: (N) `N`是`mini-batch`的大小，0 <= targets[i] <= C-1

### class torch.nn.NLLLoss(weight=None, size_average=True)[source]
负的`log likelihood loss`损失。用于训练一个`n`类分类器。

如果提供的话，`weight`参数应该是一个`1-D`tensor，里面的值对应类别的权重。当你的训练集样本不均衡的话，使用这个参数是非常有用的。

输入是一个包含类别`log-probabilities`的`2-D` tensor，形状是`（mini-batch， n）`

可以通过在最后一层加`LogSoftmax`来获得类别的`log-probabilities`。

如果您不想增加一个额外层的话，您可以使用`CrossEntropyLoss`。

此`loss`期望的`target`是类别的索引 (0 to N-1, where N = number of classes)

此`loss`可以被表示如下：
$$
loss(x, class) = -x[class]
$$
如果`weights`参数被指定的话，`loss`可以表示如下：
$$
loss(x, class) = -weights[class] * x[class]
$$
参数说明：

- weight (Tensor, optional) – 手动指定每个类别的权重。如果给定的话，必须是长度为`nclasses`

- size_average (bool, optional) – 默认情况下，会计算`mini-batch``loss`的平均值。然而，如果`size_average=False`那么将会把`mini-batch`中所有样本的`loss`累加起来。

形状:

- Input: (N,C) , `C`是类别的个数

- Target: (N) ， `target`中每个值的大小满足 `0 <= targets[i] <= C-1`

例子:
```python
 m = nn.LogSoftmax()
 loss = nn.NLLLoss()
 # input is of size nBatch x nClasses = 3 x 5
 input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
 # each element in target has to have 0 <= value < nclasses
 target = autograd.Variable(torch.LongTensor([1, 0, 4]))
 output = loss(m(input), target)
 output.backward()
```

### class torch.nn.NLLLoss2d(weight=None, size_average=True)[source]

对于图片的 `negative log likehood loss`。计算每个像素的 `NLL loss`。

参数说明：

- weight (Tensor, optional) – 用来作为每类的权重，如果提供的话，必须为`1-D`tensor，大小为`C`：类别的个数。

- size_average – 默认情况下，会计算 `mini-batch` loss均值。如果设置为 `False` 的话，将会累加`mini-batch`中所有样本的`loss`值。默认值：`True`。

形状：

- Input: (N,C,H,W)  `C` 类的数量

- Target: (N,H,W) where each value is 0 <= targets[i] <= C-1

例子：
```python
 m = nn.Conv2d(16, 32, (3, 3)).float()
 loss = nn.NLLLoss2d()
 # input is of size nBatch x nClasses x height x width
 input = autograd.Variable(torch.randn(3, 16, 10, 10))
 # each element in target has to have 0 <= value < nclasses
 target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
 output = loss(m(input), target)
 output.backward()
```
### class torch.nn.KLDivLoss(weight=None, size_average=True)[source]

计算 KL 散度损失。

KL散度常用来描述两个分布的距离，并在输出分布的空间上执行直接回归是有用的。

与`NLLLoss`一样，给定的输入应该是`log-probabilities`。然而。和`NLLLoss`不同的是，`input`不限于`2-D` tensor，因为此标准是基于`element`的。

`target` 应该和 `input`的形状相同。

此loss可以表示为：
$$
loss(x,target)=\frac{1}{n}\sum_i(target_i*(log(target_i)-x_i))
$$
默认情况下，loss会基于`element`求平均。如果 `size_average=False` `loss` 会被累加起来。

### class torch.nn.BCELoss(weight=None, size_average=True)[source]

计算 `target` 与 `output` 之间的二进制交叉熵。
$$
loss(o,t)=-\frac{1}{n}\sum_i(t[i]* log(o[i])+(1-t[i])* log(1-o[i]))
$$
如果`weight`被指定 ：
$$
loss(o,t)=-\frac{1}{n}\sum_iweights[i]* (t[i]* log(o[i])+(1-t[i])* log(1-o[i]))
$$

这个用于计算 `auto-encoder` 的 `reconstruction error`。注意 0<=target[i]<=1。

默认情况下，loss会基于`element`平均，如果`size_average=False`的话，`loss`会被累加。

### class torch.nn.MarginRankingLoss(margin=0, size_average=True)[source]

创建一个标准，给定输入 $x1$,$x2$两个1-D mini-batch Tensor's，和一个$y$(1-D mini-batch tensor) ,$y$里面的值只能是-1或1。

如果 `y=1`，代表第一个输入的值应该大于第二个输入的值，如果`y=-1`的话，则相反。

`mini-batch`中每个样本的loss的计算公式如下：

$$loss(x, y) = max(0, -y * (x1 - x2) + margin)$$

如果`size_average=True`,那么求出的`loss`将会对`mini-batch`求平均，反之，求出的`loss`会累加。默认情况下，`size_average=True`。

### class torch.nn.HingeEmbeddingLoss(size_average=True)[source]

给定一个输入 $x$(2-D mini-batch tensor)和对应的 标签 $y$ (1-D tensor,1,-1)，此函数用来计算之间的损失值。这个`loss`通常用来测量两个输入是否相似，即：使用L1 成对距离。典型是用在学习非线性 `embedding`或者半监督学习中：

$$
loss(x,y)=\frac{1}{n}\sum_i
\begin{cases}
x_i, &\text if~y_i==1 \\
max(0, margin-x_i), &if ~y_i==-1
\end{cases}
$$
$x$和$y$可以是任意形状，且都有`n`的元素，`loss`的求和操作作用在所有的元素上，然后除以`n`。如果您不想除以`n`的话，可以通过设置`size_average=False`。

`margin`的默认值为1,可以通过构造函数来设置。

### class torch.nn.MultiLabelMarginLoss(size_average=True)[source]
计算多标签分类的 `hinge loss`(`margin-based loss`) ，计算`loss`时需要两个输入： input x(`2-D mini-batch Tensor`)，和 output y(`2-D tensor`表示mini-batch中样本类别的索引)。

$$
loss(x, y) = \frac{1}{x.size(0)}\sum_{i=0,j=0}^{I,J}(max(0, 1 - (x[y[j]] - x[i])))
$$
其中 `I=x.size(0),J=y.size(0)`。对于所有的 `i`和 `j`，满足 $y[j]\neq0, i \neq y[j]$

`x` 和 `y` 必须具有同样的 `size`。

这个标准仅考虑了第一个非零 `y[j] targets`
此标准允许了，对于每个样本来说，可以有多个类别。

### class torch.nn.SmoothL1Loss(size_average=True)[source]
平滑版`L1 loss`。

loss的公式如下：
$$
loss(x, y) = \frac{1}{n}\sum_i
\begin{cases}
0.5*(x_i-y_i)^2, & if~|x_i - y_i| < 1\\
|x_i - y_i| - 0.5,  & otherwise    
\end{cases}
$$
此loss对于异常点的敏感性不如`MSELoss`，而且，在某些情况下防止了梯度爆炸，(参照 `Fast R-CNN`)。这个`loss`有时也被称为 `Huber loss`。

x 和 y 可以是任何包含`n`个元素的tensor。默认情况下，求出来的`loss`会除以`n`，可以通过设置`size_average=True`使loss累加。

### class torch.nn.SoftMarginLoss(size_average=True)[source]

创建一个标准，用来优化2分类的`logistic loss`。输入为 `x`（一个 2-D mini-batch Tensor）和 目标`y`（一个包含1或-1的Tensor）。
$$
loss(x, y) = \frac{1}{x.nelement()}\sum_i (log(1 + exp(-y[i]* x[i])))
$$
如果求出的`loss`不想被平均可以通过设置`size_average=False`。

### class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)[source]

创建一个标准，基于输入x和目标y的 `max-entropy`，优化多标签 `one-versus-all` 的损失。`x`:2-D mini-batch Tensor;`y`:binary 2D Tensor。对每个mini-batch中的样本，对应的loss为：
$$
loss(x, y) = - \frac{1}{x.nElement()}\sum_{i=0}^I y[i]\text{log}\frac{exp(x[i])}{(1 + exp(x[i])}
                      + (1-y[i])\text{log}\frac{1}{1+exp(x[i])}
$$
其中 `I=x.nElement()-1`, $y[i] \in \{0,1\}$，`y` 和 `x`必须要有同样`size`。

### class torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)[source]

给定 输入 `Tensors`，`x1`, `x2` 和一个标签Tensor `y`(元素的值为1或-1)。此标准使用`cosine`距离测量两个输入是否相似，一般用来用来学习非线性`embedding`或者半监督学习。

`margin`应该是-1到1之间的值，建议使用0到0.5。如果没有传入`margin`实参，默认值为0。

每个样本的loss是：
$$
loss(x, y) =
\begin{cases}
1 - cos(x1, x2),              &if~y ==  1 \\
max(0, cos(x1, x2) - margin), &if~y == -1
\end{cases}
$$
如果`size_average=True` 求出的loss会对batch求均值，如果`size_average=False`的话，则会累加`loss`。默认情况`size_average=True`。

### class torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)[source]
用来计算multi-class classification的hinge loss（magin-based loss）。输入是 `x`(2D mini-batch Tensor), `y`(1D Tensor)包含类别的索引， `0 <= y <= x.size(1))`。

对每个mini-batch样本：
$$
loss(x, y) = \frac{1}{x.size(0)}\sum_{i=0}^I(max(0, margin - x[y] + x[i])^p)
$$
其中 `I=x.size(0)` $i\neq y$。
可选择的，如果您不想所有的类拥有同样的权重的话，您可以通过在构造函数中传入`weights`参数来解决这个问题，`weights`是一个1D权重Tensor。

传入weights后，loss函数变为：
$$
loss(x, y) = \frac{1}{x.size(0)}\sum_imax(0, w[y] * (margin - x[y] - x[i]))^p
$$
默认情况下，求出的loss会对mini-batch取平均，可以通过设置`size_average=False`来取消取平均操作。

## Vision layers

### class torch.nn.PixelShuffle(upscale_factor)[source]

将shape为$[N, C*r^2, H, W]$的`Tensor`重新排列为shape为$[N, C, H*r, W*r]$的Tensor。
当使用`stride=1/r` 的sub-pixel卷积的时候，这个方法是非常有用的。

请看paper[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network by Shi et. al (2016)](https://arxiv.org/abs/1609.05158) 获取详细信息。

参数说明：

- upscale_factor (int) – 增加空间分辨率的因子

Shape:

- Input: $[N,C*upscale\_factor^2,H,W$]

- Output: $[N,C,H*upscale\_factor,W*upscale\_factor]$

例子:
```python
>>> ps = nn.PixelShuffle(3)
>>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
>>> output = ps(input)
>>> print(output.size())
torch.Size([1, 1, 12, 12])
```


### class torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)[source]

对于多channel 输入 进行 `2-D` 最近邻上采样。

可以通过`size`或者`scale_factor`来指定上采样后的图片大小。

当给定`size`时，`size`的值将会是输出图片的大小。

参数：

- size (tuple, optional) – 一个包含两个整数的元组 (H_out, W_out)指定了输出的长宽
- scale_factor (int, optional) – 长和宽的一个乘子

形状：

- Input: (N,C,H_in,W_in)
- Output: (N,C,H_out,W_out)  Hout=floor(H_in∗scale_factor) Wout=floor(W_in∗scale_factor)

例子：
```python
>>> inp
Variable containing:
(0 ,0 ,.,.) =
  1  2
  3  4
[torch.FloatTensor of size 1x1x2x2]

>>> m = nn.UpsamplingNearest2d(scale_factor=2)
>>> m(inp)
Variable containing:
(0 ,0 ,.,.) =
  1  1  2  2
  1  1  2  2
  3  3  4  4
  3  3  4  4
[torch.FloatTensor of size 1x1x4x4]
```

### class torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)[source]

对于多channel 输入 进行 `2-D` `bilinear` 上采样。

可以通过`size`或者`scale_factor`来指定上采样后的图片大小。

当给定`size`时，`size`的值将会是输出图片的大小。

参数：

- size (tuple, optional) – 一个包含两个整数的元组 (H_out, W_out)指定了输出的长宽
- scale_factor (int, optional) – 长和宽的一个乘子

形状：

- Input: (N,C,H_in,W_in)
- Output: (N,C,H_out,W_out)  Hout=floor(H_in∗scale_factor) Wout=floor(W_in∗scale_factor)

例子：
```python
>>> inp
Variable containing:
(0 ,0 ,.,.) =
  1  2
  3  4
[torch.FloatTensor of size 1x1x2x2]

>>> m = nn.UpsamplingBilinear2d(scale_factor=2)
>>> m(inp)
Variable containing:
(0 ,0 ,.,.) =
  1.0000  1.3333  1.6667  2.0000
  1.6667  2.0000  2.3333  2.6667
  2.3333  2.6667  3.0000  3.3333
  3.0000  3.3333  3.6667  4.0000
[torch.FloatTensor of size 1x1x4x4]
```
## Multi-GPU layers
### class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)[source]

在模块级别上实现数据并行。

此容器通过将`mini-batch`划分到不同的设备上来实现给定`module`的并行。在`forward`过程中，`module`会在每个设备上都复制一遍，每个副本都会处理部分输入。在`backward`过程中，副本上的梯度会累加到原始`module`上。

batch的大小应该大于所使用的GPU的数量。还应当是GPU个数的整数倍，这样划分出来的每一块都会有相同的样本数量。

请看: [Use nn.DataParallel instead of multiprocessing]()

除了`Tensor`，任何位置参数和关键字参数都可以传到DataParallel中。所有的变量会通过指定的`dim`来划分（默认值为0）。原始类型将会被广播，但是所有的其它类型都会被浅复制。所以如果在模型的`forward`过程中写入的话，将会被损坏。

参数说明：

- module – 要被并行的module
- device_ids – CUDA设备，默认为所有设备。
- output_device – 输出设备（默认为device_ids[0]）

例子：
```python
 net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
 output = net(input_var)
```

## Utilities
工具函数
### torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)[source]
Clips gradient norm of an iterable of parameters.

正则項的值由所有的梯度计算出来，就像他们连成一个向量一样。梯度被`in-place operation`修改。

参数说明:
- parameters (Iterable[Variable]) – 可迭代的`Variables`，它们的梯度即将被标准化。
- max_norm (float or int) – `clip`后，`gradients` p-norm 值
- norm_type (float or int) – 标准化的类型，p-norm. 可以是`inf` 代表 infinity norm.

[关于norm](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)

返回值:

所有参数的p-norm值。

### torch.nn.utils.rnn.PackedSequence(\_cls, data, batch_sizes)[source]
Holds the data and list of batch_sizes of a packed sequence.

All RNN modules accept packed sequences as inputs.
所有的`RNN`模块都接收这种被包裹后的序列作为它们的输入。

`NOTE：`
这个类的实例不能手动创建。它们只能被 `pack_padded_sequence()` 实例化。

参数说明:

- data (Variable) – 包含打包后序列的`Variable`。

- batch_sizes (list[int]) – 包含 `mini-batch` 中每个序列长度的列表。

#### torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)[source]
Packs a Variable containing padded sequences of variable length.
把一个包含 填充过的变长序列 `Variable`打包。

输入的形状可以是(T×B×* )。`T`是最长序列长度，`B`是`batch size`，`*`代表任意维度(可以是0)。如果`batch_first=True`的话，那么相应的 `input size` 就是 `(B×T×*)`。

`Variable`中保存的序列，应该按序列长度的长短排序，长的在前，短的在后。即`input[:,0]`代表的是最长的序列，`input[:, B-1]`保存的是最短的序列。

`NOTE：`
只要是维度大于等于2的`input`都可以作为这个函数的参数。你可以用它来打包`labels`，然后用`RNN`的输出和打包后的`labels`来计算`loss`。通过`PackedSequence`对象的`.data`属性可以获取 `Variable`。

参数说明:

- input (Variable) – 变长序列 被填充后的 batch

- lengths (list[int]) – `Variable` 中 每个序列的长度。

- batch_first (bool, optional) – 如果是`True`，input的形状应该是`B*T*size`。

返回值:

一个`PackedSequence` 对象。

### torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False)[source]

填充变长序列的打包的batch
Pads a packed batch of variable length sequences.
这个操作和pack_padded_sequence()是相反的。

返回的Varaible的值的`size`是 `T×B×*`, `T` 是最长序列的长度，`B` 是 batch_size,如果 `batch_first=True`,那么返回值是`B×T×*`。

Batch中的元素将会以它们长度的逆序排列。


参数说明:

- sequence (PackedSequence) – 将要被填充的 batch

- batch_first (bool, optional) – 如果为True，返回的数据的格式为 `B×T×*`。

返回值:
一个tuple，包含被填充后的序列，和batch中序列的长度列表。
