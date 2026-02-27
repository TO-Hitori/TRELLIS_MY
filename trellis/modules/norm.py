import torch
import torch.nn as nn



class LayerNorm32(nn.LayerNorm):
    '''
    在调用父类的 forward 方法之前，将输入张量 x 转换为 float32。
    执行完正则化后，它会将结果转换回输入张量的原始数据类型（x.dtype），确保输出与输入张量的精度一致。
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        # 将输入张量 x 的维度重新排列，将通道维度（dim=1）移动到最后一个维度。
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        # 将输入张量 x 的维度重新排列，将通道维度（dim=1）移动到最后一个维度。
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    