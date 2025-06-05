# Fine-Tuning

## Directory

1. [Adapter-Tuning](#adapter-tuning)
2. [LoRA-Tuning](#lora-tuning)
3. [Prefix-Tuning](#prefix-tuning)
4. [P-Tuning](#p-tuning)

## Adapter-Tuning



## LoRA-Tuning

### 2.1 背景

#### 2.1.1 方法依据

- 模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配

#### 2.1.2 假设

- 模型在任务适配过程中权重的改变量是低秩（low rank）的

### 2.2 细节

#### 2.2.1 思想

- 在原始 PLM (Pre-trained Language Model) 旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的intrinsic rank
- 训练的时候固定 PLM 的参数，只训练降维矩阵 A 与升维矩阵 B 。而模型的输入输出维度不变，输出时将 BA 与 PLM 的参数叠加
- 用随机高斯分布初始化 A ，用 0 矩阵初始化 B ，保证训练的开始此旁路矩阵依然是 0 矩阵

#### 2.2.2 应用举例

假设要在下游任务微调一个预训练语言模型（如 GPT-3），则需要更新预训练模型参数，公式表示如下
$$W_0+\Delta W$$
- $W_0$ 是预训练模型初始化的参数
- $\Delta W$ 是需要更新的参数，如果是全量微调，则参数量等于$W_0$；对于LoRA，只需要微调$\Delta W$

假设预训练矩阵为$W_0 \in \mathbb{R}^{d\times k}$，则更新可表示为：
$$W_0+\Delta W=W_0+BA,B\in \mathbb{R}^{d\times r},\ A\in \mathbb{R}^{r\times k}\ \ \ (r\ll \min (d,k))$$

- A负责将输入映射到低维空间；B从低维空间映射回原始维度
- A和B的初始化：A使用随机高斯分布初始化，B使用0矩阵初始化
- 为什么A不用0矩阵初始化：若A也用0矩阵初始化，则B矩阵梯度始终为0，无法更新参数，导致$\Delta W=0$。
  > 假设模型损失为L，梯度更新公式为
  > - $\frac{\partial L}{\partial A}=\frac{\partial L}{\partial \Delta W}\cdot B^T$
  > - $\frac{\partial L}{\partial B}=A^T\cdot \frac{\partial L}{\partial \Delta W}$
  >
  > 明显，A和B不能全都为0；若A和B全都不为0，相当于对原来的模型添加了扰动，可能导致初始性能下降或训练不稳定
  > 若A为0，B随机初始化
  > - B会因为获取0梯度而不更新
  > - 梯度会在B处消失，无法传播到A处，A也难以训练
  >
  > 若A随机，B为0 
  > - A会因为获取0梯度不更新
  > - 梯度刚开始会在A处消失，但B梯度不为0，B会继续训练，一旦为非0值，会带动A的训练
  > 
  > 但是上面的解释比较牵强，在https://arxiv.org/pdf/2406.08447 中解释到，通过实验得出，A随机允许更高的学习率，会有更好的性能


在LoRA训练过程中，$W_0$是固定的，只有A和B是训练参数
在向前过程中，$W_0$与$\Delta W$都会乘以相同的输入$x$，最后相加：
$$h=W_0x+\Delta Wx=W_0x+BAx$$
- 类似于残差连接，使用旁路更新模拟全量微调的过程，且全量微调可看作LoRA的特例（$r=k$）

在推理过程中，LoRA 也几乎未引入额外的 Inference Latency，只需要计算 $W=W_0+\Delta W$ 即可
LoRA 与 Transformer 的结合也很简单，仅在 QKV Attention 的计算中增加一个旁路


## Prefix-Tuning



## P-Tuning
