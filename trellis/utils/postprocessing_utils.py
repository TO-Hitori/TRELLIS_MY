from typing import *
import numpy as np
import torch
import utils3d
import nvdiffrast.torch as dr
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..representations import MeshExtractResult


@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    从多个视图中光栅化网格并移除不可见的面。
    同时包括后处理步骤：
        1. 移除可见性低的连接组件。
        2. 使用最小割（mincut）算法，移除通过小孔与外部相连的网格内侧面。
    Args:
        verts (torch.Tensor): 网格的顶点。形状为 (V, 3)。
        faces (torch.Tensor): 网格的面。形状为 (F, 3)。
        max_hole_size (float): 要填充的孔洞的最大面积。
        resolution (int): 光栅化的分辨率。
        num_views (int): 用于光栅化网格的视图数量。
        verbose (bool): 是否打印进度。
    """
    # 构建相机视角（yaws, pitchs为偏航角和俯仰角）
    yaws = []
    pitchs = []
    for i in range(num_views):
        # 使用哈密尔顿序列生成视角
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    # 将角度转换为张量并将其放到GPU上
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    
    # 相机的半径
    radius = 2.0
    # 视场角（40度）
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    # 计算投影矩阵
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    
    # 计算相机视角矩阵
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        # 使用look-at方法计算视图矩阵
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    # 将视角矩阵堆叠成一个大的矩阵
    views = torch.stack(views, dim=0)

    # Rasterize
    # 初始化可见性张量
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend='cuda') # 创建光栅化上下文
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        # 光栅化每个视角的三角形面片
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        # 获取可见面并更新可见性
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    # 可见性归一化
    visblity = visblity.float() / num_views
    
    # Mincut # 使用最小割算法（mincut）去除不可见面
    ## 构建外部面
    # 计算边缘信息
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    # 找到边界边
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    # 计算连接组件
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    
    # 将具有高可见性（上四分位数）的连接组件标记为外部面
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## 构建内部面（不可见的面）
    # 找到不可见的面
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0: # 如果没有不可见面，直接返回原始网格
        return verts, faces
    
    ## 构建双图（面为节点，边为边）
    # 计算双图
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    # 计算双图边的权重
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## 求解最小割问题
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])                    # 添加面节点
    g.add_edges(dual_edges.cpu().numpy())             # 添加边
    g.es['weight'] = dual_edges_weights.cpu().numpy() # 边权重
    
    ### 添加源节点（s）和目标节点（t）
    g.add_vertex('s')
    g.add_vertex('t')
    
    ### 连接不可见面到源节点（s）
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    ### 连接外部面到目标节点（t）
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
                
    ### 求解最小割
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### 检查割的有效性（检查连接组件）
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### 检查连接组件的可见性
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### 检查割的环是否足够小
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    if debug:
        # 可视化调试，生成PLY文件显示连接组件
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    # 根据有效的连接组件更新面索引，去除不需要的面
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
    
    # 使用pyMeshFix库填充小孔        
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    对网格进行后处理，包括简化网格，去除不可见面和去除孤立部分。

    参数：
        vertices (np.array): 网格的顶点。形状为 (V, 3)。
        faces (np.array): 网格的面。形状为 (F, 3)。
        simplify (bool): 是否简化网格，使用四面体边收缩算法。
        simplify_ratio (float): 简化后保留的面比例。
        fill_holes (bool): 是否填充网格中的孔洞。
        fill_holes_max_hole_size (float): 填充孔洞时，孔的最大面积。
        fill_holes_max_hole_nbe (int): 填充孔洞时，最大边界边数。
        fill_holes_resolution (int): 光栅化时的分辨率。
        fill_holes_num_views (int): 用于光栅化的视图数量。
        verbose (bool): 是否打印进度。
    """
    # 如果启用verbose，打印当前网格的顶点和面的数量
    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # 如果启用简化，并且简化比例大于0，则进行网格简化
    if simplify and simplify_ratio > 0:
        # 创建 PolyData 对象，并进行网格简化
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        # 更新顶点和面
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose: # 如果启用verbose，打印简化后的网格信息
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # 如果启用孔洞填充功能，执行孔洞填充
    if fill_holes:
        # 将顶点和面移至GPU上处理
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        # 填充孔洞
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        # 将处理后的数据移回CPU并转为NumPy数组
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:# 如果启用verbose，打印孔洞填充后的网格信息
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
    # 返回处理后的顶点和面
    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    使用xatlas将网格进行参数化映射到纹理空间。

    参数：
        vertices (np.array): 网格的顶点。形状为 (V, 3)。
        faces (np.array): 网格的面。形状为 (F, 3)。
    """
    # 使用xatlas对网格进行参数化，得到顶点映射、索引和UV坐标
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    # 根据参数化映射重新排序顶点
    vertices = vertices[vmapping]
    faces = indices # 更新面索引

    # 返回重新排序后的顶点、更新后的面和UV坐标
    return vertices, faces, uvs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
):
    """
    从多个观测图像中烘焙纹理到网格。

    参数：
        vertices (np.array): 网格的顶点。形状为 (V, 3)。
        faces (np.array): 网格的面。形状为 (F, 3)。
        uvs (np.array): 网格的UV坐标。形状为 (V, 2)。
        observations (List[np.array]): 观测图像列表。每张观测图像为二维图像，形状为 (H, W, 3)。
        masks (List[np.array]): 掩膜图像列表。每张掩膜图像为二维图像，形状为 (H, W)。
        extrinsics (List[np.array]): 外参矩阵列表。形状为 (4, 4)。
        intrinsics (List[np.array]): 内参矩阵列表。形状为 (3, 3)。
        texture_size (int): 纹理的大小。
        near (float): 相机的近裁剪平面。
        far (float): 相机的远裁剪平面。
        mode (Literal['fast', 'opt']): 纹理烘焙模式。
        lambda_tv (float): 优化中总变差（TV）损失的权重。
        verbose (bool): 是否打印进度。
    """
    # 将输入数据移至GPU进行计算
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    # 将观测图像归一化并移至GPU
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    # 创建布尔型掩膜并移至GPU
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    # 将外参和内参转换为视图矩阵和透视矩阵
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]
            
            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':# 如果选择了优化模式
        # 初始化光栅化上下文
        rastctx = utils3d.torch.RastContext(backend='cuda')
        # 翻转观察图像
        observations = [observations.flip(0) for observations in observations]
        # 翻转掩膜
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
            with torch.no_grad():
                # 获取每一张观测图像的UV坐标和UV导数
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())       # 保存UV坐标
                _uv_dr.append(rast['uv_dr'].detach()) # 保存UV导数
        # 将纹理初始化为可训练的参数
        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        # 使用Adam优化器进行纹理优化
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)
        # 定义余弦退火学习率调度函数
        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        # 定义总变差损失（防止纹理出现伪影）
        def tv_loss(texture):
            return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
    
        total_steps = 2500 # 设置优化的总步数
        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad() # 清空梯度
                # 随机选择一个视角
                selected = np.random.randint(0, len(views))
                # 选择当前视角的数据
                uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
                # 渲染当前纹理
                render = dr.texture(texture, uv, uv_dr)[0]
                # 计算渲染图像和观测图像的损失
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture) # 加入总变差损失
                loss.backward()  # 反向传播
                optimizer.step() # 更新纹理
                 # 更新学习率
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()}) # 显示当前损失
                pbar.update() # 更新进度条
        # 获取最终纹理并进行填补
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        # 使用OpenCV进行填补
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        # 如果模式不正确，抛出错误
        raise ValueError(f'Unknown mode: {mode}')
    # 返回最终烘焙的纹理
    return texture


def to_glb(
    app_rep: MeshExtractResult,
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    将生成的资产转换为glb文件。

    参数：
        app_rep (Union[Strivec, Gaussian]): 外观表示方法，表示材质的特征。
        mesh (MeshExtractResult): 提取的网格数据。
        simplify (float): 简化比例，用于简化网格时删除的面数的比例。
        fill_holes (bool): 是否填充网格中的孔洞。
        fill_holes_max_size (float): 填充的孔洞的最大面积。
        texture_size (int): 纹理的大小。
        debug (bool): 是否打印调试信息。
        verbose (bool): 是否打印处理过程中的进度。
    """
    # 获取网格的顶点和面，并将其转为NumPy数组
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # 网格后处理：包括网格简化和孔洞填充
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,       # 如果简化比例大于0，启用简化
        simplify_ratio=simplify,     # 设置简化比例
        fill_holes=fill_holes,       # 是否填充孔洞
        fill_holes_max_hole_size=fill_holes_max_size,   # 设置最大孔洞面积
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)), # 根据简化比例调整孔洞边界数量
        fill_holes_resolution=1024,  # 设置孔洞填充的分辨率
        fill_holes_num_views=1000,   # 设置视图数
        debug=debug,                 # 是否启用调试模式
        verbose=verbose,
    )
    print("-to_glb postprocess_mesh success")

    # 对网格进行参数化处理，生成UV映射
    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    print("-to_glb parametrize_mesh success")

    # 渲染多视角图像来烘焙纹理
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    print("-to_glb render_multiview success")
    
    # 创建掩膜：在每张观测图像中，非零区域即为有效区域
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    # 转换外参和内参为NumPy数组
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    # 烘焙纹理，使用优化模式
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    print("-to_glb bake_texture success")
    texture = Image.fromarray(texture) # 将纹理数据转换为图像对象

     # 旋转网格：将Z轴向上的网格旋转为Y轴向上的网格
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # 创建PBR材质，并应用到网格上
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    print("-to_glb  trimesh.visual.material.PBRMaterial success")
    # 创建Trimesh对象，设置网格的顶点、面和材质
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    print("-to_glb success")
    return mesh
