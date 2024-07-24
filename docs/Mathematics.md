# Mathematics

## Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

## ReLU

$$
max(0,x)
$$

## Cross Entropy

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
