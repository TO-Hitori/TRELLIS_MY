import torch
'''
函数：construct_dense_grid
功能：
- 根据给定的分辨率和设备生成稠密的立方体网格。
- 计算每个体素（立方体）的顶点坐标，并返回顶点和角点索引。
- 主要用于构建立方体网格，供后续特征聚合和数据处理使用。

函数：construct_voxel_grid
功能：
- 将体素网格的坐标（coords）转换为唯一顶点坐标，并返回每个体素的唯一顶点索引。
- 通过去重，减少计算量并提高处理效率。
- 返回唯一顶点的坐标和每个体素的角点索引，为后续特征聚合和损失计算提供基础。

函数：cubes_to_verts
功能：
- 将体素数据中的特征（value）按照体素的角点索引聚合到顶点上。
- 使用 `scatter_reduce` 方法对角点特征进行聚合，支持不同的聚合方式（如均值、总和等）。
- 为每个顶点提供聚合后的特征数据，通常用于训练或可视化。

函数：sparse_cube2verts
功能：
- 通过 `construct_voxel_grid` 和 `cubes_to_verts` 将稀疏体素数据的特征聚合到顶点上。
- 如果在训练模式下，计算并返回重构损失（即体素特征与聚合顶点特征之间的均方误差）。
- 返回聚合后的顶点坐标、新特征以及损失值，通常用于训练神经网络模型。

函数：get_dense_attrs
功能：
- 根据给定的体素坐标（coords）和特征（feats），填充一个稠密的网格属性张量（dense_attrs）。
- 支持初始化 SDF 外部区域的值，适用于体积数据的处理和可视化。
- 将体素数据转换为稠密网格表示，便于后续操作，如优化和渲染。

函数：get_deformed_verts
功能：
- 通过变形参数（deform）对网格顶点的位置进行调整。
- 使用 `tanh` 函数平滑变形，确保变形不会过大，适用于形变模拟和网格动画。
- 返回变形后的顶点坐标，通常用于动态网格和动画效果。
'''
# 单位立方体的8个角点坐标
'''
        6[0,1,1] ──────────── 7[1,1,1]
       ╱ │                   ╱│
      ╱  │                  ╱ │
     ╱   │                 ╱  │
    4[0,0,1]───────────── 5[1,0,1]
    │    │               │    │
    │    │               │    │   
    │    2[0,1,0] ───────┼────3 [1,1,0]
    │   ╱                │   ╱
    │  ╱                 │  ╱
    │ ╱                  │ ╱
    0[0,0,0]──────────── 1[1,0,0]
          
          z ↑
            │ ↗y
            └─────→ x          
          
          6─────────[6]──────────7
         ╱│                     ╱│
        ╱ │                    ╱ │
     [11] │                 [10] │
      ╱  [7]                 ╱  [5]
     ╱    │                 ╱    │
    4─────┼────[2]─────────5     │
    │     │                │     │
    │     2─────────[4]────┼─────3
    │    ╱                 │    ╱
   [3]  ╱                 [1]  ╱
    │ [8]                  │ [9]
    │ ╱                    │ ╱
    │╱                     │╱
    0──────────[0]─────────1
                  
'''
cube_corners = torch.tensor([[0, 0, 0], 
                             [1, 0, 0], 
                             [0, 1, 0], 
                             [1, 1, 0], 
                             [0, 0, 1], 
                             [1, 0, 1], 
                             [0, 1, 1], 
                             [1, 1, 1]], dtype=torch.int)
# 立方体在3D空间中的6个邻域偏移（正负x、y、z方向），用于查找相邻立方体。
cube_neighbor = torch.tensor([[+1, 0, 0], 
                              [-1, 0, 0], 
                              [0, +1, 0], 
                              [0, -1, 0], 
                              [0, 0, +1], 
                              [0, 0, -1]])
# 表示立方体的12条边（连接的顶点索引）
cube_edges = torch.tensor(
    [0, 1,   # 0
     1, 5,   # 1
     4, 5,   # 2
     0, 4,   # 3
     2, 3,   # 4
     3, 7,   # 5
     6, 7,   # 6
     2, 6,   # 7
     2, 0,   # 8
     3, 1,   # 9
     7, 5,   # 10
     6, 4],  # 11
    dtype=torch.long, requires_grad=False)
    
# 构建稠密网格 
def construct_dense_grid(res, device='cuda'):
    '''res表示网格的分辨率'''
    # 每个边的点数量
    res_v = res + 1
    # 包含了整个网格中的每个顶点的索引：0 到 res_v^3 - 1
    vertsid = torch.arange(res_v ** 3, device=device)
    
    '''
    1. 将 vertsid 转换成一个三维张量，形状为 (res_v, res_v, res_v)
    2. 通过切片，只保留前 res 行、列和层。忽略最后一层和行。
    3. 将三维张量展平成一维张量，得到每个顶点的索引。
    '''
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    
    '''
    cube_corners_bias 是一个包含了每个立方体的 8 个角点相对坐标的偏移量，用于在后续的操作中计算每个体素的 8 个角点的位置。
        - cube_corners[:, 0] * res_v：首先根据角点的第一个坐标（x坐标），乘以 res_v 来获得 x 轴偏移。
        - cube_corners[:, 1]：然后加上角点的第二个坐标（y坐标）来得到 y 轴偏移。
        - res_v：再乘以 res_v 来调整偏移量，使得每一层的偏移都能正确计算。
        - cube_corners[:, 2]：最后加上角点的第三个坐标（z坐标）来得到最终的偏移量。
    '''
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    
    '''
    cube_fx8 的形状是 (n, 8)，表示每个体素（或网格点）对应的 8 个角点的全局索引。
        - coordsid.unsqueeze(1)：将 coordsid 在第一个维度上增加一个维度，使其变成一个列向量（形状为 (n, 1)，其中 n 是网格中点的数量）。这样做是为了方便与 cube_corners_bias 进行逐元素相加。
        - cube_corners_bias.unsqueeze(0).to(device)：将 cube_corners_bias 在第一个维度上增加一个维度，变成形状为 (1, 8)，并将它移动到指定的设备（如 GPU）。这是为了进行广播。
        - (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0))：通过广播机制，将每个坐标点的索引加上每个角点的偏移量，从而得到每个体素的 8 个角点的全局索引。
    '''
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))

    '''
    verts 是一个包含网格中每个顶点位置的张量，形状为 (n, 3)，表示每个顶点的三维坐标。
        - vertsid // (res_v ** 2)：通过整除 res_v^2 来获得每个点的 x 坐标（在每一层中每行有 res_v^2 个点）。
        - (vertsid // res_v) % res_v：首先通过整除 res_v 获得 y 坐标，然后通过对 res_v 取模来获得每行中的 y 坐标。
        - vertsid % res_v：对 vertsid 取模得到 z 坐标。
    '''
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)

    return verts, cube_fx8
    
# 根据给定的坐标（coords）构建一个体素网格
def construct_voxel_grid(coords):
    '''coords 是一个包含体素坐标的张量，形状为 (n, 3)，表示 n 个体素的坐标'''
    '''
    计算每个体素（立方体）中 8 个角点的全球坐标
        - 将 cube_corners 的形状从 (8, 3) 变为 (1, 8, 3)
        - .unsqueeze(1): 变成形状为 (n, 1, 3)
        - 通过广播机制，将每个体素的坐标与 8 个角点的偏移坐标相加，得到每个体素的 8 个角点的全局坐标。(n, 8, 3) 的张量，表示 n 个体素每个角点的全球坐标。
        - 将形状为 (n, 8, 3) 的张量展平为形状 (n*8, 3)   
    '''
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    
    '''
    - torch.unique 函数用于返回一个去重后的张量（verts_unique），并且通过 return_inverse=True 还会返回一个 inverse_indices 张量，它表示每个原始顶点在去重后的唯一顶点中的位置。
    - verts_unique：去除了重复的顶点坐标后的张量，形状为 (unique_n, 3)，其中 unique_n 是唯一顶点的数量
    - inverse_indices：表示原始顶点索引在唯一顶点中的位置，形状为 (n*8)
    '''
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    
    # cubes 是一个形状为 (n, 8) 的张量，表示 n 个体素的 8 个角点在唯一顶点中的索引
    cubes = inverse_indices.reshape(-1, 8)
    
    return verts_unique, cubes

# 将体素（立方体）中的多个特征值按其角点坐标聚合到对应的顶点上
def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    '''
    num_verts：顶点的数量，表示网格中所有唯一顶点的数量。
    cubes：形状为 (n, 8) 的张量，表示 n 个体素的 8 个角点在唯一顶点坐标中的索引。
    value：形状为 (n, 8, M) 的张量，其中 M 是特征的维度。它表示 n 个体素每个角点的特征数据（例如，颜色、密度等）。
    reduce：聚合操作，默认值是 'mean'，表示将相同顶点的特征值按平均值进行聚合。其他可能的聚合操作有 'sum'、'max' 等。
    '''
    M = value.shape[2] # 角点的特征数
    # 创建了一个大小为 (num_verts, M) 的张量 reduced，用于存储每个顶点的聚合特征值。
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    
    # scatter_reduce 将根据 cubes 中的角点索引将 value 中的特征聚合到 reduced 张量中。最终返回的是 reduced，它包含了每个顶点的聚合特征。
    return torch.scatter_reduce(
        reduced, 
        0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), 
        reduce=reduce, 
        include_self=False
    )
    '''
    cubes_to_verts 函数将每个体素（立方体）中的 8 个角点的特征数据聚合到网格顶点上，具体做法是：
        根据每个体素的角点索引，将角点的特征数据展平。
        使用 scatter_reduce 将每个体素的角点特征聚合到对应的顶点位置。
        最终返回一个包含每个顶点聚合特征的张量。
    '''
    
# 将稀疏体素数据（例如，体素网格中某些位置的特征）聚合到网格顶点    
def sparse_cube2verts(coords, feats, training=True):
    '''
    coords：输入的体素坐标，形状通常为 (n, 3)，其中 n 是体素的数量，表示每个体素在三维网格中的位置。
    feats：输入的特征，形状通常为 (n, M)，表示每个体素的特征数据（例如，颜色、密度等），M 是特征的维度。
    training：一个布尔值，指示是否在训练模式下。如果为 True，函数将计算重构损失；如果为 False，不会计算损失。
    '''
    
    '''
    new_coords：是体素网格中所有唯一顶点的坐标。
    cubes：是一个表示每个体素8个角点在唯一顶点中的索引的张量，形状为 (n, 8)。
    '''
    new_coords, cubes = construct_voxel_grid(coords)
    
    '''
    将体素的 8 个角点的特征值聚合成对应顶点的特征
    返回的 new_feats 形状为 (num_verts, M)，其中 num_verts 是唯一顶点的数量，M 是每个顶点的特征维度。
    '''
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    

def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    '''
    coords：表示坐标的张量，形状为 (n, 3)，其中 n 是体素的数量，表示每个体素的三维位置。
    feats：表示特征的张量，形状为 (n, F)，其中 F 是特征的维度，表示每个体素的特征数据。
    res：表示网格的分辨率，即每个坐标轴上的体素数量。
    sdf_init：一个布尔值，指示是否初始化 SDF（Signed Distance Field）值。如果为 True，初始化时设置 SDF 外部区域的初始值。
    '''
    # 获取特征的通道数 F
    F = feats.shape[-1]
    
    '''
    创建了一个形状为 (res, res, res, F) 的稠密属性张量 dense_attrs，并将其初始化为零。
    dense_attrs 用来存储网格中每个体素的特征：
        [res] * 3：表示三维网格的尺寸（在每个坐标轴上都有 res 个体素）。
        [F]：表示每个体素有 F 个特征。
    '''
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    if sdf_init:
        dense_attrs[..., 0] = 1 # initial outside sdf value
    '''
    这行代码将 feats 中的特征数据填充到对应的体素坐标位置：
        coords[:, 0], coords[:, 1], coords[:, 2]：分别提取出 coords 中的三个坐标维度（x, y, z），并将它们作为索引，用来将 feats 中的数据放置到 dense_attrs 的相应位置。
        dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats：将 feats 中的特征值赋值给对应位置的 dense_attrs。
    '''
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    '''
    将 dense_attrs 张量的形状从 (res, res, res, F) 重塑为 (res^3, F)，
    即将三维网格的所有体素展开成一维。返回的张量形状为 (res^3, F)，
    表示每个体素在网格中的特征。
    '''
    return dense_attrs.reshape(-1, F)


def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    '''
    v_pos：表示顶点位置的张量，形状为 (n, 3)，其中 n 是顶点的数量，表示每个顶点的三维坐标。
    deform：表示变形信息的张量，形状为 (n, )，包含每个顶点的变形参数。
    res：表示网格的分辨率，用于归一化变形。
    '''
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)
    '''
    这行代码的作用是根据给定的变形信息 deform 计算变形后的顶点位置：
        v_pos / res：将顶点的坐标 v_pos 进行归一化，除以 res 将顶点位置调整到 [0, 1] 范围内。
        - 0.5：将位置调整到 [-0.5, 0.5] 范围内。这是为了确保网格的中心对称。
        (1 - 1e-8) / (res * 2)：这是一个非常小的常数，避免除以零，并将变形的幅度调整到合适的范围。
        torch.tanh(deform)：使用 tanh 函数对变形进行平滑处理。deform 是变形参数，通过 tanh 函数进行非线性调整，确保变形不会过大。
    '''
        

if __name__ == "__main__":
    print("cube_corners", cube_corners.shape)
    print("cube_neighbor", cube_neighbor.shape)
    print("cube_edges", cube_edges.shape)