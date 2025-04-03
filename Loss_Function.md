# Loss Function
## MSE Loss (Mean Squared Error)
> 回归任务（关键点检测、图像重建）

### 公式
> $$L=\frac{1}{N}\sum^N_{i=1}(y_i-\widehat{y}_i)^2$$

### 代码
```python
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(axis=None)

# 反向传播梯度计算（用于手动更新权重）
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

torch.nn.MSELoss(reduction="mean") # "none" "mean"(default) "sum"
```

### 优缺点
优点：
* 处处可导
* 优化稳定
  
缺点：
* 异常值敏感
* 分类问题梯度更新效率低

## Cross-Entropy Loss

## Focal Loss
$$FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)$$
