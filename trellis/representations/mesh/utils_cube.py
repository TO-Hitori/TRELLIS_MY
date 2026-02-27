import torch

# 定义了一个单位立方体的8个顶点的相对坐标
cube_corners = torch.tensor([[0, 0, 0], 
                             [1, 0, 0], 
                             [0, 1, 0], 
                             [1, 1, 0], 
                             [0, 0, 1], 
                             [1, 0, 1], 
                             [0, 1, 1], 
                             [1, 1, 1]], dtype=torch.int)
# 定义了三维空间中一个点的6个直接邻居的偏移量（上、下、左、右、前、后）。
cube_neighbor = torch.tensor([[+1, 0, 0], 
                              [-1, 0, 0], 
                              [0, +1, 0], 
                              [0, -1, 0], 
                              [0, 0, +1], 
                              [0, 0, -1]])
# 定义了立方体12条边的连接关系，通过连接8个顶点的索引来表示。
cube_edges = torch.tensor([0, 1, 
                           1, 5, 
                           4, 5, 
                           0, 4, 
                           2, 3, 
                           3, 7, 
                           6, 7, 
                           2, 6,
                           2, 0, 
                           3, 1, 
                           7, 5, 
                           6, 4], dtype=torch.long, requires_grad=False)
    
# 此函数根据给定的分辨率 res 构建一个密集的 3D 网格。     
def construct_dense_grid(res, device='cuda'):
    '''
    为一个给定分辨率 res 的三维空间，构建一个完整的、密集的体素网格（Voxel Grid）的拓扑结构。
    construct a dense grid based on resolution
    '''
    # 点数
    res_v = res + 1
    # 为每一个顶点分配一个独一无二的ID
    vertsid = torch.arange(res_v ** 3, device=device)
    # 目的: 找出每一个小立方体的“原点”顶点ID。
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    # 计算从一个立方体的“原点”顶点到它其他7个角的一维ID偏移量。
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    # 构建最终的立方体-顶点映射表
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    # 将一维的顶点ID vertsid 转换回三维坐标。
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    # 返回计算好的顶点坐标和立方体-顶点映射表。
    return verts, cube_fx8
    '''
    verts
    tensor([[0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2]], device='cuda:0')
            
    cube_fx8
    tensor([[ 0,  9,  3, 12,  1, 10,  4, 13],
            [ 1, 10,  4, 13,  2, 11,  5, 14],
            [ 3, 12,  6, 15,  4, 13,  7, 16],
            [ 4, 13,  7, 16,  5, 14,  8, 17],
            [ 9, 18, 12, 21, 10, 19, 13, 22],
            [10, 19, 13, 22, 11, 20, 14, 23],
            [12, 21, 15, 24, 13, 22, 16, 25],
            [13, 22, 16, 25, 14, 23, 17, 26]], device='cuda:0')
            
    '''




def construct_voxel_grid(coords):
    '''
    根据给定的稀疏体素坐标，构建一个稀疏的体素网格的拓扑结构。
    '''
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    M = value.shape[2] # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)
    
    
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    

def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    F = feats.shape[-1]
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    if sdf_init:
        dense_attrs[..., 0] = 1 # initial outside sdf value
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    return dense_attrs.reshape(-1, F)


def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)
        