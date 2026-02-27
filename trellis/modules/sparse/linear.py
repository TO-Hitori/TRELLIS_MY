import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseLinear'
]
'''
SparseLinear 类是一个专为稀疏张量（SparseTensor）设计的线性层。
    它继承了 PyTorch 的 nn.Linear 层，并重写了 forward 方法，使得输入为稀疏张量时，能够保持稀疏格式进行高效的计算。
    输入稀疏张量的特征数据（feats）被提取出来，并通过标准的 nn.Linear 层进行线性变换。
    变换后的结果被重新包装为一个稀疏张量，保持了原始的坐标和其他元数据。
'''

class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
