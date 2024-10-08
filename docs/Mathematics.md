# Mathematics

## Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

## ReLU

$$
max(0,x)
$$

## Softmax

$$
\hat{y}_j = \frac{\exp(z_j)}{\sum_{k=1}^{C} \exp(z_k)}
$$

## Standardization（标准化）

标准化是将数据变换为均值为0、标准差为1的分布。这种方法适用于数据呈高斯分布或接近高斯分布的情况。

$$
z = \frac{x - \mu}{\sigma}
$$

- $x$ 是原始数据
- $μ$ 是数据的均值
- $σ$ 是数据的标准差
- $z$ 是标准化后的数据

## Normalization（归一化）

将数据缩放到 [0, 1] 范围内:

$$
x' = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$

将数据缩放到 [-1, 1] 范围内:

$$
x' = 2 \cdot \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}} - 1
$$

- $x$ 是原始数据
- $x_{\text{min}}$ 是数据的最小值
- $x_{\text{max}}$ 是数据的最大值
- $x'$ 是归一化后的数据

## Loss

常见损失函数

### L1 Loss

平均绝对值误差（MAE）

### L2 Loss

平均方差（MSE）

### Binary Cross Entropy Loss

用于二分类问题，即每个样本只有两个类别（例如，0和1），使用 sigmoid 激活函数使输出的概率在 0 ~ 1

$$
\text{BCE}(y, \hat{y}) = -\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

- $y_i$ 是样本 i 的真实标签（0 或 1）
- $\hat{y}_i$ 是模型对样本 i 的预测概率（需要经过sigmoid处理，表示预测的标签为1的概率）

### Categorical Cross Entropy Loss (Softmax Loss)

用于多分类问题，计算 loss 前需要使用 Softmax 使得预测的概率和为1

$$
\text{CCE}(y, \hat{y}) = -\sum_{j=1}^{C} y_j \log(\hat{y}_j)
$$

- $y_j$ 是实际的概率（属于 $j$ 类则为1，否则为0）
- $\hat{y}_j$ 是预测为 $j$类的概率
  
该 loss 实际上只与预测为真实类别的概率有关
