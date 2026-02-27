import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        """
        3D网格提取结果，用于存储网格的顶点、面、顶点属性等信息。

        参数:
            vertices (torch.Tensor): 网格的顶点坐标。
            faces (torch.Tensor): 网格的面索引。
            vertex_attrs (torch.Tensor, optional): 顶点属性（如颜色、法线等），默认值为None。
            res (int, optional): 网格的分辨率，默认值为64。
        """
        # 存储网格的顶点
        self.vertices = vertices
        # 存储网格的面索引，转换为长整型
        self.faces = faces.long()
        # 存储顶点属性
        self.vertex_attrs = vertex_attrs
        # 计算每个面的法线
        self.face_normal = self.comput_face_normals(vertices, faces)
        # 网格的分辨率
        self.res = res
        # 如果顶点和面不为空，设置为成功
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # 用于训练时的变量，仅在训练中使用
        self.tsdf_v = None    # 体素距离场（TSDF）值
        self.tsdf_s = None    # 体素距离场（TSDF）符号
        self.reg_loss = None  # 正则化损失
        
    def comput_face_normals(self, verts, faces):
        """
        计算网格面片的法线。
        
        参数:
            verts (torch.Tensor): 顶点坐标。
            faces (torch.Tensor): 面索引。
        
        返回:
            torch.Tensor: 每个面的法线，形状为 [num_faces, 3]。
        """
        i0 = faces[..., 0].long() # 获取每个面的第一个顶点的索引
        i1 = faces[..., 1].long() # 获取每个面的第二个顶点的索引
        i2 = faces[..., 2].long() # 获取每个面的第三个顶点的索引

        v0 = verts[i0, :] # 获取第一个顶点的坐标
        v1 = verts[i1, :] # 获取第二个顶点的坐标
        v2 = verts[i2, :] # 获取第三个顶点的坐标
        # 使用叉积计算面法线 
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        # 对法线进行归一化
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # 返回法线，形状为 [num_faces, 3, 1]，重复以便与面片一致
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        """
        计算顶点法线，通过加权平均每个面法线来计算每个顶点的法线。
        
        参数:
            verts (torch.Tensor): 顶点坐标。
            faces (torch.Tensor): 面索引。
        
        返回:
            torch.Tensor: 顶点法线，形状为 [num_vertices, 3]。
        """
        i0 = faces[..., 0].long() # 获取每个面的第一个顶点的索引
        i1 = faces[..., 1].long() # 获取每个面的第二个顶点的索引
        i2 = faces[..., 2].long() # 获取每个面的第三个顶点的索引

        v0 = verts[i0, :]  # 获取第一个顶点的坐标
        v1 = verts[i1, :]  # 获取第二个顶点的坐标
        v2 = verts[i2, :]  # 获取第三个顶点的坐标
        
        # 使用叉积计算每个面的法线
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        # 初始化一个与顶点数相同的零张量，用于存储顶点法线
        v_normals = torch.zeros_like(verts)
        # 将面法线加到对应的顶点上
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)
        
        # 对法线进行归一化
        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        # 返回顶点法线
        return v_normals   


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        """
        从稀疏体素特征生成网格的模型，使用FlexiCube进行计算。

        参数:
            device (str): 计算设备（如"cuda"）。
            res (int): 网格的分辨率，默认值为64。
            use_color (bool): 是否使用颜色特征，默认为True。
        """
        super().__init__()
        self.device=device
        self.res = res
        # 初始化FlexiCubes，用于网格提取
        self.mesh_extractor = FlexiCubes(device=device)
        # 初始化SDF偏差
        self.sdf_bias = -1.0 / res
        # 构建密集网格
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        # 设置是否使用颜色特征
        self.use_color = use_color
        # 计算特征布局
        self._calc_layout()
    
    def _calc_layout(self):
        """
        计算特征的布局，定义了每种特征的形状和大小。
        """
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},       # SDF特征的布局
            'deform': {'shape': (8, 3), 'size': 8 * 3},# 形变特征的布局
            'weights': {'shape': (21,), 'size': 21}    # 权重特征的布局
        }
        if self.use_color:
            '''
            # 如果使用颜色特征，则添加颜色特征的布局
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS) # 将布局存储为EasyDict
        start = 0
        for k, v in self.layouts.items():
            # 为每个特征分配范围
            v['range'] = (start, start + v['size'])
            start += v['size']
        # 计算总的特征通道数
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        """
        获取指定特征布局的特征数据。

        参数:
            feats (torch.Tensor): 特征数据。
            name (str): 特征名称。
        
        返回:
            torch.Tensor: 根据特征名称获取的特征数据。
        """
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        根据指定的稀疏体素结构生成网格。
        Args:
            cube_attrs [Nx21] : 包含稀疏体素特征的输入数据。
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # 添加SDF偏差到特征中
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        # 获取不同特征的布局
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias # 调整SDF值
        # 选择是否包含颜色特征
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        
        # 获取稠密属性
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        
        # 根据颜色特征选择是否进行颜色处理
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
        
        # 计算形变后的顶点    
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        # 使用FlexiCubes生成网格
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
        
        # 生成网格提取结果
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        # 如果是训练模式，计算训练损失
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
