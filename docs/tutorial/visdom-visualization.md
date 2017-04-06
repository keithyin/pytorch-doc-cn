
# **Visdom** PyTorch可视化工具

**本文翻译的时候把 略去了 `Torch`部分。**

[项目地址](https://github.com/facebookresearch/visdom)

![visdom_big](https://lh3.googleusercontent.com/-bqH9UXCw-BE/WL2UsdrrbAI/AAAAAAAAnYc/emrxwCmnrW4_CLTyyUttB0SYRJ-i4CCiQCLcB/s0/Screen+Shot+2017-03-06+at+10.51.02+AM.png"visdom_big")

一个灵活的可视化工具，可用来对于 实时，富数据的 创建，组织和共享。支持`Torch`和`Numpy`。

* [总览](#总览)
* [基本概念](#基本概念)
* [Setup](#setup)
* [启动](#启动)
* [可视化接口](#可视化接口)
* [总结](#总结)


## 总览

`Visdom`目的是促进**远程**数据的可视化，重点是支持科学实验。。


<p align="center"><img src="https://lh3.googleusercontent.com/-h3HuvbU2V0SfgqgXGiK3LPghE5vqvS0pzpObS0YgG_LABMFk62JCa3KVu_2NV_4LJKaAa5-tg=s0" width="500"  /></p>

向您和您的合作者发送可视化 图像，图片和文本。

<p align="center"><img src="https://thumbs.gfycat.com/SlipperySecondhandGemsbuck-size_restricted.gif" width="500" /></p>

通过编程组织您的可视化空间，或者通过`UI`为实时数据创建`dashboards`，检查实验的结果，或者`debug`实验代码。

<p align="center"><img align="center" src="https://lh3.googleusercontent.com/-IHexvZ-FMtk/WLTXBgQlijI/AAAAAAAAm_s/514LM8R1XFgyNKPVMf4tNwYluZsHsC63wCLcB/s0/Screen+Shot+2017-02-27+at+3.15.27+PM.png" width="500" /></p>



 <br/>

## 基本概念
`Visdom`有一组简单的特性，可以用它们组合成不同的用例。


#### Panes（窗格）
<p align="center"><img align="center" src="https://lh3.googleusercontent.com/-kLnogsg9RCs/WLx34PEsGWI/AAAAAAAAnSs/7t_62pbfmfoEBnkcbKTXIqz0WM8pQJHVQCLcB/s0/Screen+Shot+2017-03-05+at+3.34.43+PM.png" width="500" /></p>

`UI`刚开始是个白板--您可以用图像，图片，文本填充它。这些填充的数据出现在 `Panes` 中，您可以这些`Panes`进行 拖放，删除，调整大小和销毁操作。`Panes`是保存在 `envs` 中的， `envs`的状态 存储在会话之间。您可以下载`Panes`中的内容--包括您在`svg`中的绘图。



> **Tip**: 您可以使用浏览器的放大缩小功能来调整UI的大小。



#### Environments(环境)
<p align="center"><img align="center" src="https://lh3.googleusercontent.com/-1wRSpNIoFeo/WLXacodRTMI/AAAAAAAAnEo/sTr5jSnQviA0uLqFIvwPGledmxcpupdkgCLcB/s0/Screen+Shot+2017-02-28+at+2.54.13+PM.png" width="300" /></p>

您可以使用`envs`对可视化空间进行分区。默认地，每个用户都会有一个叫做`main`的`envs`。可以通过编程或`UI`创建新的`envs`。`envs`的状态是长期保存的。

您可以通过 url: `http://localhost.com:8097/env/main`访问特定的`env`。
You can access a specific env via url: `http://localhost.com:8097/env/main`. 如果您的服务器是被托管的，那么您可以将此`url`分享给其他人，那么其他人也会看到您的可视化结果。

>**管理 Envs:**
>在初始化服务器的时候，您的 envs 默认通过`$HOME/.visdom/` 加载。您也可以将自定义的路径 当作命令行参数 传入。如果您移除了Env文件夹下的`.json`文件，那么相应的环境也会被删除。

#### State（状态）
一旦您创建了一些可视化，状态是被保存的。服务器自动缓存您的可视化--如果您重新加载网页，您的可视化会重新出现。

<p align="center"><img align="center" src="https://lh3.googleusercontent.com/-ZKeFJfMe5S4/WLXebiNgFwI/AAAAAAAAnFI/AH2cGsf40hEWbH6UeclYQcZPS0YZbcayQCLcB/s0/env_fork_2.gif" width="400" /></p>

* **Save:** 你可以手动的保存`env`通过点击`save`按钮。它会首先序列化`env`的状态，然后以`json`文件的形式保存到硬盘上，包括窗口的位置。 同样，您也可以通过编程来实现`env`的保存。
<br/>当面对一些十分复杂的可视化，例如参数设置非常重要，这中保存`env`状态的方法是十分有用的。例：数据丰富的演示，模型的训练 `dashboard`， 或者 系统实验。这种设计依旧可以使这些可视化十分容易分享和复用。


* **Fork:** 有过您输入了一个新的`env` 名字，`saving`会建立一个心的`env` -- 有效的**forking**之前的状态。（注：这个fork等价于github的fork，跟复制的意思差不多）


## Setup

需要 Python 2.7/3 (and optionally Torch7)

```bash
# Install Python server and client，如果您使用python的话，装这一个就可以了。
pip install visdom

```

## 启动

启动服务器（可能在`screen`或者`tmux`中）：

```bash
python -m visdom.server
```

一旦启动服务器，您就可以通过在浏览器中输入`http://localhost:8097`来访问 `Visdom`，`localhost`可以换成您的托管地址。


>If the above does not work, try using an SSH tunnel to your server by adding the following line to your local  `~/.ssh/config`:
```LocalForward 127.0.0.1:8097 127.0.0.1:8097```.

#### Python example
```python
import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))
```

### Demos

```bash
python example/demo.py
```


## 可视化接口
`Visdom`支持下列`API`。由[Plotly](https://plot.ly/)提供可视化支持。

- `vis.scatter`  : 2D 或 3D 散点图
- `vis.line`     : 线图
- `vis.stem`     : 茎叶图
- `vis.heatmap`  : 热力图
- `vis.bar`      : 条形图
- `vis.histogram`: 直方图
- `vis.boxplot`  : 箱型图
- `vis.surf`     : 表面图
- `vis.contour`  : 轮廓图
- `vis.quiver`   : 绘出二维矢量场
- `vis.image`    : 图片
- `vis.text`     : 文本
- `vis.mesh`     : 网格图
- `vis.save`     : 序列化状态

关于上述`API`更详尽的解释将在下面给出。为了对`visdom`的能力有一个快速的了解，您可以看一下   [example](https://github.com/facebookresearch/visdom/tree/master/example) ，或者，您可以继续往下看。

这些`API`的确切输入类型有所不同，尽管大多数`API` 的输入包含，一个tensor `X`（保存数据）和一个可选的tensor `Y`（保存标签或者时间戳）。所有的绘图函数都接收一个可选参数`win`，用来将图画到一个特定的`window`上。每个绘图函数也会返回当前绘图的`win`。您也可以指定 汇出的图添加到哪个`env`上。
（这里的window的意思就是之前说的Pane）。

![visdom_big](https://lh3.googleusercontent.com/-bqH9UXCw-BE/WL2UsdrrbAI/AAAAAAAAnYc/emrxwCmnrW4_CLTyyUttB0SYRJ-i4CCiQCLcB/s0/Screen+Shot+2017-03-06+at+10.51.02+AM.png"visdom_big")

#### plot.scatter
这个函数是用来画`2D`或`3D`数据的散点图。它需要输入 `N*2`或`N*3`的 tensor `X`来指定`N`个点的位置。一个可供选择的长度为`N`的`vector`用来保存`X`中的点对应的标签(1 到 K)。 -- 标签可以通过点的颜色反应出来。

`scatter()`支持下列的选项：

- `options.colormap`    : 色图（控制图的颜色） (`string`; default = `'Viridis'`)
- `options.markersymbol`: 标记符号 (`string`; default = `'dot'`)
- `options.markersize`  : 标记大小(`number`; default = `'10'`)
- `options.markercolor` : 每个标记的颜色. (`torch.*Tensor`; default = `nil`)
- `options.legend`      : 包含图例名字的`table`

`options.markercolor` 是一个包含整数值的`Tensor`。`Tensor`的形状可以是 `N` 或 `N x 3` 或 `K` 或 `K x 3`.

- Tensor of size `N`: 表示每个点的单通道颜色强度。 0 = black, 255 = red
- Tensor of size `N x 3`: 用三通道表示每个点的颜色。 0,0,0 = black, 255,255,255 = white
- Tensor of size `K` and `K x 3`: 为每个类别指定颜色，不是为每个点指定颜色。


#### plot.line
这个函数用来画 线图。它需要一个形状为`N`或者`N×M`的tensor `Y`，用来指定 `M`条线的值(每条线上有`N`个点)。和一个可供选择的 tensor `X` 用来指定对应的 x轴的值; `X`可以是一个长度为`N`的tensor（这种情况下，M条线共享同一个 x轴），也可以是形状和`Y`一样的tensor。

The following `options` are supported:

- `options.fillarea`    : 填充线下面的区域 (`boolean`)
- `options.colormap`    : 色图 (`string`; default = `'Viridis'`)
- `options.markers`     : 显示点标记 (`boolean`; default = `false`)
- `options.markersymbol`: 标记的形状 (`string`; default = `'dot'`)
- `options.markersize`  : 标记的大小 (`number`; default = `'10'`)
- `options.legend`      : 保存图例名字的 `table`

#### plot.stem
这个函数用来画茎叶图。它需要一个 形状为`N`或者`N*M`的 tensor `X` 来指定`M`时间序列中`N`个点的值。一个可选择的`Y`，形状为`N`或者`N×M`，用`Y`来指定时间戳，如果`Y`的形状是`N`，那么默认`M`时间序列共享同一个时间戳。

支持以下特定选项：

- `options.colormap`: colormap (`string`; default = `'Viridis'`)
- `options.legend`  : `table` containing legend names

#### plot.heatmap
这个函数用来画热力图。它输入一个 形状为`N×M`的 tensor `X`。`X`指定了热力图中位置的值。

支持下列特定选项：

- `options.colormap`   : 色图 (`string`; default = `'Viridis'`)
- `options.xmin`       : 小于这个值的会被剪切成这个值(`number`; default = `X:min()`)
- `options.xmax`       : 大于这个值的会被剪切成这个值 (`number`; default = `X:max()`)
- `options.columnnames`: 包含x轴标签的`table`
- `options.rownames`   : 包含y轴标签的`table`

#### plot.bar
* [条形图wiki](https://zh.wikipedia.org/wiki/%E6%9D%A1%E5%BD%A2%E7%BB%9F%E8%AE%A1%E5%9B%BE)

这个函数可以画 正常的，堆起来的，或分组的的条形图。
输入参数：

- X(tensor):形状 `N` 或 `N×M`，指定每个条的高度。如果`X`有`M`列，那么每行的值可以看作一组或者把他们值堆起来（取决与`options.stacked`是否为True）。
- Y(tensor, optional):形状 `N`，指定对应的x轴的值。

支持以下特定选项：

- `options.columnnames`: `table` containing x-axis labels
- `options.stacked`    : stack multiple columns in `X`
- `options.legend`     : `table` containing legend labels

#### plot.histogram
* [直方图wiki](https://zh.wikipedia.org/wiki/%E7%9B%B4%E6%96%B9%E5%9B%BE)

这个函数用来画指定数据的直方图。他需要输入长度为 `N` 的 tensor `X`。`X`保存了构建直方图的值。

支持下面特定选项：

- `options.numbins`: `bins`的个数 (`number`; default = 30)

#### plot.boxplot
* [箱型图wiki](https://zh.wikipedia.org/wiki/%E7%AE%B1%E5%BD%A2%E5%9C%96)

这个函数用来画箱型图：

输入：

- X(tensor): 形状 `N`或`N×M`，指定做第`m`个箱型图的`N`个值。

支持以下特定选项：

- `options.legend`: labels for each of the columns in `X`

#### plot.surf
这个函数用来画表面图：
输入：

- X(tensor):形状 `N×M`，指定表面图上位置的值.

支持以下特定选项：

- `options.colormap`: colormap (`string`; default = `'Viridis'`)
- `options.xmin`    : clip minimum value (`number`; default = `X:min()`)
- `options.xmax`    : clip maximum value (`number`; default = `X:max()`)

#### plot.contour
这个函数用来画轮廓图。

输入：

- X(tensor)：形状 `N×M`，指定了轮廓图中的值

支持以下特定选项：

- `options.colormap`: colormap (`string`; default = `'Viridis'`)
- `options.xmin`    : clip minimum value (`number`; default = `X:min()`)
- `options.xmax`    : clip maximum value (`number`; default = `X:max()`)

#### plot.quiver
这个函数用来画二维矢量场图。

输入：

- X(tensor): 形状 `N*M`
- Y(tensor):形状 `N*M`
- gridX(tensor, optional):形状 `N*M`
- gradY(tensor, optional): 形状 `N*M`
`X` 与 `Y`决定了 箭头的长度和方向。可选的`gridX`和`gridY`指定了偏移。

支持下列特定选项：

- `options.normalize`:  最长肩头的长度 (`number`)
- `options.arrowheads`: 是否现实箭头 (`boolean`; default = `true`)

#### plot.image
这个函数用来画 图片。
输入：

- img(tensor): shape(`C*H*W`)。

支持下面特定选项:

- `options.jpgquality`: JPG quality (`number` 0-100; default = 100)

### plot.video
这个函数 播放一个 `video`。
输入： `video` 的文件名，或者是一个 shape 为`L*H*W*C` 的 `tensor`。这个函数不支持其它特定的功能选项。

注意:使用`tensor`作为输入的时候，需要安装`ffmpeg`。
能不能播放`video`取决你使用的浏览器：浏览器必须要支持`Theano codec in an OGG container`。（chrome可以用）。

### plot.svg

此函数绘制一个`SVG`对象。输入是一个`SVG`字符串或 一个`SVG`文件的名称。该功能不支持任何特定的功能
`options`。

#### plot.text
此函数可在文本框中打印文本。输入输入一个`text`字符串。目前不支持特定的`options`

### plot.mesh
此函数画出一个网格图。

输入：

- X(tensor): shape(`N*2`或`N*3`) 定义`N`个顶点

- Y(tensor， optional)：shape(`M*2`或`M×3`) 定义多边形


支持下列特定选项:

- `options.color`: color (`string`)
- `options.opacity`: 多边形的不透明性 (`number` between 0 and 1)

### Customizing plots

绘图函数使用可选的`options`表作为输入。用它来修改默认的绘图属性。所有输入参数在单个表中指定;输入参数是基于输入表中键的匹配。

下列的选项除了对于`plot.img`和`plot.txt`不可用以外，其他的绘图函数都适用。我们称他为 通用选项。

- `options.title`       : figure title
- `options.width`       : figure width
- `options.height`      : figure height
- `options.showlegend`  : show legend (`true` or `false`)
- `options.xtype`       : type of x-axis (`'linear'` or `'log'`)
- `options.xlabel`      : label of x-axis
- `options.xtick`       : show ticks on x-axis (`boolean`)
- `options.xtickmin`    : first tick on x-axis (`number`)
- `options.xtickmax`    : last tick on x-axis (`number`)
- `options.xtickstep`   : distances between ticks on x-axis (`number`)
- `options.ytype`       : type of y-axis (`'linear'` or `'log'`)
- `options.ylabel`      : label of y-axis
- `options.ytick`       : show ticks on y-axis (`boolean`)
- `options.ytickmin`    : first tick on y-axis (`number`)
- `options.ytickmax`    : last tick on y-axis (`number`)
- `options.ytickstep`   : distances between ticks on y-axis (`number`)
- `options.marginleft`  : left margin (in pixels)
- `options.marginright` : right margin (in pixels)
- `options.margintop`   : top margin (in pixels)
- `options.marginbottom`: bottom margin (in pixels)


其它的一些选项就是函数特定的选项，在上面API介绍的时候已经提到过。

## 总结
明确几个名词：

- env：看作一个大容器
- pane： 就是用于绘图的小窗口，在代码中叫 `window`

使用`Visdom`就是在`env`中的`pane`上画图。
