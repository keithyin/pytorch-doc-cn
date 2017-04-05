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

- input_size – The number of expected features in the input x
- hidden_size – The number of features in the hidden state h
- num_layers – Number of recurrent layers.
- bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
- batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
- dropout – If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
- bidirectional – If True, becomes a bidirectional RNN. Default: False

`LSTM`输入:
input, (h_0, c_0)

- input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
- h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
- c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

`LSTM`输出
output, (h_n, c_n)
- output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
- h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
- c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len

`LSTM`模型参数:
- weight_ih_l[k] – the learnable input-hidden weights of the k-th layer (W_ii|W_if|W_ig|W_io), of shape (input_size x 4*hidden_size)
- weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer (W_hi|W_hf|W_hg|W_ho), of shape (hidden_size x 4*hidden_size)
- bias_ih_l[k] – the learnable input-hidden bias of the k-th layer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)
- bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer (W_hi|W_hf|W_hg|b_ho), of shape (4*hidden_size)
示例:
```python
lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))
```

### class torch.nn.GRU(* args, ** kwargs)[source]
Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

For each element in the input sequence, each layer computes the following function:

rt=sigmoid(Wirxt+bir+Whrh(t−1)+bhr)it=sigmoid(Wiixt+bii+Whih(t−1)+bhi)nt=tanh(Winxt+bin+rt∗(Whnh(t−1)+bhn))ht=(1−it)∗nt+it∗h(t−1)
rt=sigmoid(Wirxt+bir+Whrh(t−1)+bhr)it=sigmoid(Wiixt+bii+Whih(t−1)+bhi)nt=tanh⁡(Winxt+bin+rt∗(Whnh(t−1)+bhn))ht=(1−it)∗nt+it∗h(t−1)
where htht is the hidden state at time t, xtxt is the hidden state of the previous layer at time t or inputtinputt for the first layer, and rtrt, itit, ntnt are the reset, input, and new gates, respectively.

Parameters:
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
num_layers – Number of recurrent layers.
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
dropout – If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
bidirectional – If True, becomes a bidirectional RNN. Default: False
Inputs: input, h_0
input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
Outputs: output, h_n
output (seq_len, batch, hidden_size * num_directions): tensor containing the output features h_t from the last layer of the RNN, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
Variables:
weight_ih_l[k] – the learnable input-hidden weights of the k-th layer (W_ir|W_ii|W_in), of shape (input_size x 3*hidden_size)
weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer (W_hr|W_hi|W_hn), of shape (hidden_size x 3*hidden_size)
bias_ih_l[k] – the learnable input-hidden bias of the k-th layer (b_ir|b_ii|b_in), of shape (3*hidden_size)
bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer (W_hr|W_hi|W_hn), of shape (3*hidden_size)
Examples:
```python
>>> rnn = nn.GRU(10, 20, 2)
>>> input = Variable(torch.randn(5, 3, 10))
>>> h0 = Variable(torch.randn(2, 3, 20))
>>> output, hn = rnn(input, h0)
```

### class torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')[source]
An Elman RNN cell with tanh or ReLU non-linearity.

h′=tanh(wih∗x+bih+whh∗h+bhh)
h′=tanh⁡(wih∗x+bih+whh∗h+bhh)
If nonlinearity=’relu’, then ReLU is used in place of tanh.

Parameters:
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
nonlinearity – The non-linearity to use [‘tanh’|’relu’]. Default: ‘tanh’
Inputs: input, hidden
input (batch, input_size): tensor containing input features
hidden (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
Outputs: h’
h’ (batch, hidden_size): tensor containing the next hidden state for each element in the batch
Variables:
weight_ih – the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh – the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih – the learnable input-hidden bias, of shape (hidden_size)
bias_hh – the learnable hidden-hidden bias, of shape (hidden_size)
Examples:
```python
>>> rnn = nn.RNNCell(10, 20)
>>> input = Variable(torch.randn(6, 3, 10))
>>> hx = Variable(torch.randn(3, 20))
>>> output = []
>>> for i in range(6):
...     hx = rnn(input[i], hx)
...     output.append(hx)
```

### class torch.nn.LSTMCell(input_size, hidden_size, bias=True)[source]
A long short-term memory (LSTM) cell.

i=sigmoid(Wiix+bii+Whih+bhi)f=sigmoid(Wifx+bif+Whfh+bhf)g=tanh(Wigx+big+Whch+bhg)o=sigmoid(Wiox+bio+Whoh+bho)c′=f∗c+i∗gh′=o∗tanh(ct)
i=sigmoid(Wiix+bii+Whih+bhi)f=sigmoid(Wifx+bif+Whfh+bhf)g=tanh⁡(Wigx+big+Whch+bhg)o=sigmoid(Wiox+bio+Whoh+bho)c′=f∗c+i∗gh′=o∗tanh⁡(ct)
Parameters:
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
Inputs: input, (h_0, c_0)
input (batch, input_size): tensor containing input features
h_0 (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
c_0 (batch. hidden_size): tensor containing the initial cell state for each element in the batch.
Outputs: h_1, c_1
h_1 (batch, hidden_size): tensor containing the next hidden state for each element in the batch
c_1 (batch, hidden_size): tensor containing the next cell state for each element in the batch
Variables:
weight_ih – the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh – the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih – the learnable input-hidden bias, of shape (hidden_size)
bias_hh – the learnable hidden-hidden bias, of shape (hidden_size)
Examples:
```python
>>> rnn = nn.LSTMCell(10, 20)
>>> input = Variable(torch.randn(6, 3, 10))
>>> hx = Variable(torch.randn(3, 20))
>>> cx = Variable(torch.randn(3, 20))
>>> output = []
>>> for i in range(6):
...     hx, cx = rnn(input[i], (hx, cx))
...     output.append(hx)
```

### class torch.nn.GRUCell(input_size, hidden_size, bias=True)[source]
A gated recurrent unit (GRU) cell

r=sigmoid(Wirx+bir+Whrh+bhr)i=sigmoid(Wiix+bii+Whih+bhi)n=tanh(Winx+bin+r∗(Whnh+bhn))h′=(1−i)∗n+i∗h
r=sigmoid(Wirx+bir+Whrh+bhr)i=sigmoid(Wiix+bii+Whih+bhi)n=tanh⁡(Winx+bin+r∗(Whnh+bhn))h′=(1−i)∗n+i∗h
Parameters:
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
Inputs: input, hidden
input (batch, input_size): tensor containing input features
hidden (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
Outputs: h’
h’: (batch, hidden_size): tensor containing the next hidden state for each element in the batch
Variables:
weight_ih – the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh – the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih – the learnable input-hidden bias, of shape (hidden_size)
bias_hh – the learnable hidden-hidden bias, of shape (hidden_size)
Examples:

>>> rnn = nn.GRUCell(10, 20)
>>> input = Variable(torch.randn(6, 3, 10))
>>> hx = Variable(torch.randn(3, 20))
>>> output = []
>>> for i in range(6):
...     hx = rnn(input[i], hx)
...     output.append(hx)
## Linear layers

## Dropout layers

## Sparse layers

## Distance functions

## Loss functions


## Vision layers

## Multi-GPU layers
### class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)[source]
Implements data parallelism at the module level.

This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension. In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.

The batch size should be larger than the number of GPUs used. It should also be an integer multiple of the number of GPUs so that each chunk is the same size (so that each GPU processes the same number of samples).

See also: Use nn.DataParallel instead of multiprocessing

Arbitrary positional and keyword inputs are allowed to be passed into DataParallel EXCEPT Tensors. All variables will be scattered on dim specified (default 0). Primitive types will be broadcasted, but all other types will be a shallow copy and can be corrupted if written to in the model’s forward pass.

Parameters:
module – module to be parallelized
device_ids – CUDA devices (default: all devices)
output_device – device location of output (default: device_ids[0])
Example:
```python
>>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
>>> output = net(input_var)
```
## Utilities
### torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)[source]
Clips gradient norm of an iterable of parameters.

The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

Parameters:
parameters (Iterable[Variable]) – an iterable of Variables that will have gradients normalized
max_norm (float or int) – max norm of the gradients
norm_type (float or int) – type of the used p-norm. Can be 'inf' for infinity norm.
Returns:
Total norm of the parameters (viewed as a single vector).

### torch.nn.utils.rnn.PackedSequence(\_cls, data, batch_sizes)[source]
Holds the data and list of batch_sizes of a packed sequence.

All RNN modules accept packed sequences as inputs.

Note

Instances of this class should never be created manually. They are meant to be instantiated by functions like pack_padded_sequence().

Variables:
data (Variable) – Variable containing packed sequence
batch_sizes (list[int]) – list of integers holding information about the batch size at each sequence step
#### torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)[source]
Packs a Variable containing padded sequences of variable length.

Input can be of size TxBx* where T is the length of the longest sequence (equal to lengths[0]), B is the batch size, and * is any number of dimensions (including 0). If batch_first is True BxTx* inputs are expected.

The sequences should be sorted by length in a decreasing order, i.e. input[:,0] should be the longest sequence, and input[:,B-1] the shortest one.

Note

This function accept any input that has at least two dimensions. You can apply it to pack the labels, and use the output of the RNN with them to compute the loss directly. A Variable can be retrieved from a PackedSequence object by accessing its .data attribute.

Parameters:
input (Variable) – padded batch of variable length sequences.
lengths (list[int]) – list of sequences lengths of each batch element.
batch_first (bool, optional) – if True, the input is expected in BxTx* format.
Returns:
a PackedSequence object

### torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False)[source]
Pads a packed batch of variable length sequences.

It is an inverse operation to pack_padded_sequence().

The returned Variable’s data will be of size TxBx*, where T is the length of the longest sequence and B is the batch size. If batch_size is True, the data will be transposed into BxTx* format.

Batch elements will be ordered decreasingly by their length.

Parameters:
sequence (PackedSequence) – batch to pad
batch_first (bool, optional) – if True, the output will be in BxTx* format.
Returns:
Tuple of Variable containing the padded sequence, and a list of lengths of each sequence in the batch.
