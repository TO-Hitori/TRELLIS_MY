import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import MeshRenderer
from ..representations import MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    """
    从给定的偏航角(yaw)、俯仰角(pitch)、半径(r)和视场角(fov)计算外参和内参。
    参数:
        yaws (list or float): 偏航角(yaw)，可为列表或单个值。
        pitchs (list or float): 俯仰角(pitch)，可为列表或单个值。
        rs (list or float): 半径(r)，相机与目标物体之间的距离。
        fovs (list or float): 视场角(fov)，表示相机的视野宽度。
    返回:
        extrinsics (list or tensor): 外参矩阵。
        intrinsics (list or tensor): 内参矩阵。
    """
    # 检查传入的yaws是否是列表
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    
    extrinsics = [] # 存储外参
    intrinsics = [] # 存储内参
    
    # 计算每一组相机的外参和内参
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        # 将视场角转换为弧度
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()      # 将偏航角转换为tensor
        pitch = torch.tensor(float(pitch)).cuda()  # 将俯仰角转换为tensor
        
        # 计算相机原点位置（根据偏航角、俯仰角和半径）
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r # 计算相机的3D空间位置
        
        # 使用utils3d中的函数计算外参：从相机原点到目标位置（[0,0,0]）
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        # 使用utils3d中的函数计算内参：根据视场角生成相机的内参矩阵
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

        # 存储外参和内参
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    """
    根据给定的样本获取渲染器实例。
    参数:
        sample: 输入样本，通常为MeshExtractResult类型。
    返回:
        renderer: 渲染器实例。
    """
    if isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer() # 创建MeshRenderer实例
        renderer.rendering_options.resolution = kwargs.get('resolution', 512) # 设置渲染分辨率，默认值为512
        renderer.rendering_options.near = kwargs.get('near', 1)               # 设置相机的近裁剪平面，默认值为1
        renderer.rendering_options.far = kwargs.get('far', 100)               # 设置相机的远裁剪平面，默认值为100
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)               # 设置超级采样抗锯齿，默认值为4
    else: # 如果样本类型不支持，抛出错误
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    """
    渲染多个相机视角的图像。
    参数:
        sample: 输入样本，通常为MeshExtractResult类型。
        extrinsics: 外参矩阵列表。
        intrinsics: 内参矩阵列表。
        options: 渲染参数字典。
        colors_overwrite: 用于覆盖颜色的选项（可选）。
        verbose: 是否显示进度条。
    返回:
        rets: 渲染结果字典，包括颜色、深度等。
    """
    print("Type of Render is", type(sample))
    
    # 创建渲染器
    if isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {} # 存储渲染结果的字典
    # 遍历内参和外参
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        res = renderer.render(sample, extr, intr)
        if 'normal' not in rets: rets['normal'] = []
        rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        if 'color' not in rets: rets['color'] = []
        rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            
    return rets

def render_multiview(sample, resolution=512, nviews=30):
    """
    渲染多视角图像。
    使用球面哈密尔顿序列生成多个视角的偏航角和俯仰角，并渲染相应图像。
    参数:
        sample: 输入样本，通常为MeshExtractResult类型。
        resolution: 渲染分辨率。
        nviews: 渲染的视角数量。
    返回:
        color: 渲染结果的颜色图像。
        extrinsics: 外参矩阵。
        intrinsics: 内参矩阵。
    """
    r = 2     # 设置相机半径
    fov = 40  # 设置视场角
    # 使用哈密尔顿序列生成多个视角
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]   # 提取偏航角
    pitchs = [cam[1] for cam in cams] # 提取俯仰角
    
    # 计算外参和内参
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


# def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
#     yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
#     yaw_offset = offset[0]
#     yaw = [y + yaw_offset for y in yaw]
#     pitch = [offset[1] for _ in range(4)]
#     extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
#     return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


# def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
#     yaws = torch.linspace(0, 2 * 3.1415, num_frames)
#     pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
#     yaws = yaws.tolist()
#     pitch = pitch.tolist()
#     extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
#     return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
