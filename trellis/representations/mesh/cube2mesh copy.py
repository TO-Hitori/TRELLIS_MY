import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes

'''
MeshExtractResult 类:
- 用于存储从稀疏体素数据生成的网格数据。
- 存储的信息包括：顶点坐标（vertices）、面索引（faces）、顶点附加特征（vertex_attrs）、面法线（face_normal）等。
- 提供了计算面法线（comput_face_normals）和顶点法线（comput_v_normals）的方法。
- 训练过程中还包含了体素的符号距离场（TSDF）值（tsdf_v）、SDF值（tsdf_s）和正则化损失（reg_loss）。
- 如果网格生成成功（顶点和面不为空），`success` 属性为 `True`。

SparseFeatures2Mesh 类:
- 用于从稀疏体素结构生成网格，主要通过 `FlexiCubes` 进行体素到网格的转换。
- 该类接受稀疏体素特征（如 SDF、变形、颜色、权重等），将其映射到网格的顶点。
- 支持在训练过程中计算损失，以优化生成的网格。
- 包括网格的生成步骤，首先提取特征（SDF、变形、颜色、权重等），然后使用 `sparse_cube2verts` 聚合特征并生成稠密网格。
- 最终，调用 `FlexiCubes` 提取网格顶点、面和颜色信息，并返回一个包含这些信息的 `MeshExtractResult` 对象。
- 支持通过 `training` 参数调整是否计算损失，并返回相关的正则化损失和 TSDF 值。
'''

# MeshExtractResult 类用于存储从稀疏体素数据生成的网格数据
class MeshExtractResult:
    def __init__(self,
                 vertices,       # 网格顶点
                 faces,          # 网格面
                 vertex_attrs=None,  # 顶点附加特征（如颜色、法线等）
                 res=64):        # 网格分辨率
        self.vertices = vertices         # 顶点坐标
        self.faces = faces.long()        # 面的索引，转换为长整型
        self.vertex_attrs = vertex_attrs  # 顶点的附加特征
        self.face_normal = self.comput_face_normals(vertices, faces)  # 计算面法线
        self.res = res                    # 网格分辨率
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)  # 成功标志，如果顶点和面都不为空则成功

        # 训练时用到的变量
        self.tsdf_v = None  # 用于存储体素的 TSDF（符号距离场）值
        self.tsdf_s = None  # 用于存储体素的符号距离场（SDF）值
        self.reg_loss = None  # 正则化损失

    # 计算面法线
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()  # 面的第一个顶点索引
        i1 = faces[..., 1].long()  # 面的第二个顶点索引
        i2 = faces[..., 2].long()  # 面的第三个顶点索引

        v0 = verts[i0, :]  # 获取面第一个顶点的坐标
        v1 = verts[i1, :]  # 获取面第二个顶点的坐标
        v2 = verts[i2, :]  # 获取面第三个顶点的坐标

        # 计算面法线
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # 叉积计算法线
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # 对法线进行归一化

        return face_normals[:, None, :].repeat(1, 3, 1)  # 将法线重复3次以符合面结构要求

    # 计算顶点法线
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]

        # 计算面法线
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)  # 初始化顶点法线

        # 累加与面相关的法线
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)  # 对顶点法线进行归一化
        return v_normals  # 返回计算出的顶点法线


# SparseFeatures2Mesh 类用于从稀疏体素数据生成网格，使用 FlexiCubes 进行体素到网格的转换
class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        用于从稀疏体素特征生成网格的模型，使用 FlexiCube 结构
        '''
        super().__init__()
        self.device = device  # 计算设备（默认为cuda）
        self.res = res  # 网格分辨率
        self.mesh_extractor = FlexiCubes(device=device)  # 初始化 FlexiCubes 对象
        self.sdf_bias = -1.0 / res  # SDF 偏移值，用于调整 SDF 范围
        verts, cube = construct_dense_grid(self.res, self.device)  # 构建稠密网格
        self.reg_c = cube.to(self.device)  # 立方体索引
        self.reg_v = verts.to(self.device)  # 顶点坐标
        self.use_color = use_color  # 是否使用颜色特征
        self._calc_layout()  # 计算特征布局

    # 计算并存储特征布局
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},  # SDF 特征，8个角点，1个通道
            'deform': {'shape': (8, 3), 'size': 8 * 3},  # 变形特征，8个角点，3个通道
            'weights': {'shape': (21,), 'size': 21}  # 权重特征，21个通道
        }
        if self.use_color:
            # 如果使用颜色特征，增加6通道颜色特征
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)  # 将布局信息存入字典
        start = 0
        # 计算每个特征的索引范围
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])  # 每个特征的索引范围
            start += v['size']
        self.feats_channels = start  # 计算总特征通道数

    # 获取特定布局的特征数据
    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        """
        从稀疏体素数据生成网格。
        参数:
            cubefeats [Nx21]: 稀疏体素特征，包括体素的权重信息。
            verts_attrs [Nx10]: 顶点特征，包括 SDF、变形、颜色、法线等。
        返回:
            返回网格信息，包括成功标志和损失。
        """
        # 获取体素的坐标和特征
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        # 获取各类特征：SDF、变形、颜色和权重
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        
        # 对 SDF 进行偏移处理
        sdf += self.sdf_bias
        
        # 根据是否使用颜色特征决定需要的顶点特征
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        
        # 将体素特征聚合到顶点
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        
        # 获取稠密网格的特征
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        
        # 如果使用颜色特征，提取颜色
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        # 计算变形后的顶点位置
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        # 使用 FlexiCubes 提取网格
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        # 创建网格提取结果
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        
        # 训练时计算损失
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)  # 计算 TSDF
            mesh.tsdf_s = v_attrs[:, 0]  # 计算 SDF
        return mesh  # 返回网格提取结果