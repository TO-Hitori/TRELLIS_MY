import numpy as np
"""
该代码实现了生成低差异序列的功能，主要包括Halton序列、Hammersley序列以及球面坐标上的低差异采样。
这些序列广泛应用于蒙特卡罗模拟、数值积分等领域，能够提供均匀分布的随机样本。

功能概述：
1. **Halton序列**：
   - 使用质数作为基数生成低差异序列。该序列适用于多维采样，能够生成均匀分布的点。
   - `halton_sequence(dim, n)` 函数生成维度为`dim`，样本索引为`n`的Halton序列。

2. **Hammersley序列**：
   - 通过归一化的样本索引和Halton序列的组合生成Hammersley序列。Hammersley序列常用于蒙特卡罗模拟中的低差异序列。
   - `hammersley_sequence(dim, n, num_samples)` 函数生成维度为`dim`，样本索引为`n`的Hammersley序列。

3. **球面上的Hammersley序列**：
   - 生成球面坐标系中的经度（phi）和纬度（theta），用于球面上的低差异随机采样。该函数支持偏移量调整，并能够对`u`进行映射，改变其分布方式。
   - `sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False)` 函数生成球面坐标中的低差异点。

这些序列可以用于图形学中的随机采样、数值积分、计算机仿真等领域，特别适用于需要均匀分布的随机点的场景。
"""



# 定义常量：前16个质数，用于生成Halton序列的基数
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

# 反向基数函数，用于计算基数转换
def radical_inverse(base, n):
    val = 0                 # 初始化结果值
    inv_base = 1.0 / base   # 计算基数的倒数
    inv_base_n = inv_base   # 初始化倒数基数
    
    # 循环处理n的每一位
    while n > 0:
        digit = n % base           # 取出当前位的数字
        val += digit * inv_base_n  # 将当前位数字乘以倒数基数并累加
        n //= base                 # 处理下一位，除去当前位
        inv_base_n *= inv_base     # 更新倒数基数
    return val                     # 返回计算后的反向基数值

# 生成Halton序列的函数
def halton_sequence(dim, n):
    # 使用质数基数来生成反向基数，并返回每一维的结果
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

# 生成Hammersley序列的函数
def hammersley_sequence(dim, n, num_samples):
    # 生成第一维，归一化索引n/num_samples，然后加上Halton序列的其他维度
    return [n / num_samples] + halton_sequence(dim - 1, n)

# 生成球面上的Hammersley序列，返回球面坐标（经度、纬度）
def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    # 获取二维Hammersley序列作为u和v
    u, v = hammersley_sequence(2, n, num_samples)
    
    # 根据偏移量调整u和v的值
    u += offset[0] / num_samples # 对u应用偏移，确保归一化
    v += offset[1]               # 对v应用偏移
    
    # 如果remap为True，则对u进行额外映射
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3 # 对u做线性映射
    
    # 计算纬度theta（从u得到），并将u映射到球面坐标
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    
    # 计算经度phi（从v得到），并将v映射到[0, 2π]范围内
    phi = v * 2 * np.pi
    
    # 返回球面坐标（经度、纬度）
    return [phi, theta]