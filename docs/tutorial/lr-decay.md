# pytorch learning rate decay
本文主要是介绍在`pytorch`中如何使用`learning rate decay`.
先上代码:
```python
def adjust_learning_rate(optimizer, epoch):
    """
    每50个epoch,权重以0.99的速率衰减
    """
    if epoch // 50 == 0:
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.99
```
**什么是`param_groups`?**
`optimizer`通过`param_group`来管理参数组.`param_group`中保存了参数组及其对应的学习率,动量等等.所以我们可以通过更改`param_group['lr']`的值来更改对应参数组的学习率.

```python
# 有两个`param_group`即,len(optim.param_groups)==2
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```
