import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F
"""
该文件实现了一个网格渲染器类 `MeshRenderer`，用于渲染3D网格对象，并返回多个类型的渲染结果。
通过该渲染器，我们可以生成包括颜色图、深度图、法线图、法线贴图和遮罩图等多种渲染输出。

主要功能：
1. **`intrinsics_to_projection`**：将OpenCV的相机内参矩阵转换为OpenGL的透视投影矩阵。该矩阵可用于将3D世界坐标投影到2D图像平面。
   - 输入：相机内参矩阵、近平面距离、远平面距离。
   - 输出：OpenGL透视矩阵。

2. **`MeshRenderer` 类**：
   - **`__init__`**：初始化渲染器，设置渲染参数如分辨率、近平面、远平面、超级采样抗锯齿等。创建CUDA/OpenGL的渲染上下文。
   - **`render`**：渲染网格模型。该方法根据传入的相机外参（extrinsics）和内参（intrinsics），计算渲染结果，并返回指定类型的图像（如颜色图、法线图、深度图等）。
     - 输入：网格模型（MeshExtractResult），相机的外参和内参矩阵，渲染结果类型列表（如"mask", "normal", "depth", "color"）。
     - 输出：包含不同渲染结果的字典，如颜色图、法线图、深度图等。

3. **渲染过程**：
   - 使用`nvdiffrast`光栅化库进行渲染，支持在GPU上进行高效的渲染计算。
   - 渲染过程中支持抗锯齿处理，并根据需要进行超级采样抗锯齿（SSAA）。
   - 支持根据相机内外参计算投影矩阵，并将网格从世界坐标系转换到相机坐标系和裁剪坐标系。

4. **支持的渲染结果**：
   - **`mask`**：生成遮罩图，标记有效区域。
   - **`depth`**：生成深度图，表示相机到物体的距离。
   - **`normal`**：生成法线图，表示网格表面每个点的法向量。
   - **`normal_map`**：生成法线贴图，通常用于纹理映射。
   - **`color`**：生成颜色图，表示网格的颜色信息。

5. **超级采样抗锯齿（SSAA）**：
   - 如果启用了SSAA（`ssaa > 1`），则在更高的分辨率下渲染图像，然后通过插值缩放回目标分辨率，以提高图像质量，减少锯齿现象。

6. **`yaw_pitch_r_fov_to_extrinsics_intrinsics`**：
   - 将偏航角、俯仰角、半径和视场角（fov）转换为相机的外参和内参矩阵，常用于生成不同视角的渲染结果。

该文件主要用于3D网格渲染，广泛应用于计算机图形学、虚拟现实、增强现实、3D重建和可视化等领域。
"""

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV内参转换为OpenGL透视矩阵。

    参数:
        intrinsics (torch.Tensor): [3, 3] OpenCV内参矩阵。
        near (float): 近平面距离。
        far (float): 远平面距离。

    返回:
        (torch.Tensor): [4, 4] OpenGL透视矩阵。
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1] # 焦距（像素单位）
    cx, cy = intrinsics[0, 2], intrinsics[1, 2] # 主点（通常接近图像中心）
    # 创建一个4x4的零矩阵，用于存储透视矩阵
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    
    # 设置透视矩阵的x和y方向的缩放因子
    ret[0, 0] = 2 * fx # x 方向的缩放因子。
    ret[1, 1] = 2 * fy # y 方向的缩放因子。
    
    # 设置透视矩阵的x和y方向的偏移
    ret[0, 2] = 2 * cx - 1   # x 方向的偏移
    ret[1, 2] = - 2 * cy + 1 # y 方向的偏移
    
    # 设置透视矩阵的深度相关值
    ret[2, 2] = far / (far - near)          # 远近平面的比值
    ret[2, 3] = near * far / (near - far)   # 近平面和远平面之间的距离比值
    ret[3, 2] = 1.                          # 设置透视矩阵的z分量
    return ret


class MeshRenderer:
    """
    Mesh表示的渲染器。

    参数:
        rendering_options (dict): 渲染选项。
        glctx (nvdiffrast.torch.RasterizeGLContext): 用于CUDA/OpenGL交互的RasterizeGLContext对象。
    """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,  # 渲染分辨率
            "near": None,        # 近平面 
            "far": None,         # 远平面
            "ssaa": 1            # 超级采样抗锯齿倍数
        })
        self.rendering_options.update(rendering_options)
        # 创建RasterizeCudaContext对象，用于处理CUDA/OpenGL交互
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth", "color"]
        ) -> edict:
        """
        渲染网格。

        Args:
            mesh :  输入网格（MeshExtractResult）。
            extrinsics (torch.Tensor): (4, 4) 相机外参矩阵。
            intrinsics (torch.Tensor): (3, 3) 相机内参矩阵。
            return_types (list): 渲染结果类型的列表，包括 "mask", "normal", "depth", "color" 等。

        Returns:包含渲染结果的字典
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        # 获取渲染选项的参数
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        # 检查mesh是否有效（即有顶点和面）
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            # 如果mesh无效，则返回全黑图像
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        # 计算透视矩阵
        perspective = intrinsics_to_projection(intrinsics, near, far)
        # 将相机外参矩阵进行扩展
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)

        # 获取网格的顶点并进行齐次坐标转换
        vertices = mesh.vertices.unsqueeze(0)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)

        # 计算相机空间下的顶点
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))

        # 计算裁剪空间下的顶点
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        
        # 获取面索引
        faces_int = mesh.faces.int()

        # 使用nvdiffrast进行光栅化，计算栅格图
        rast, _ = dr.rasterize(
            glctx=self.glctx, 
            pos=vertices_clip, 
            tri=faces_int, 
            resolution=(resolution * ssaa, resolution * ssaa)
        )
        # 存储渲染结果
        out_dict = edict()
        
        # 根据返回类型计算不同的渲染结果
        for type in return_types:
            img = None
            if type == "mask" :
                # 使用光栅化结果中的深度值来生成遮罩图，深度大于0的地方为有效区域
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                # 从相机坐标中提取深度信息，并对光栅化结果进行抗锯齿处理
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                # 插值计算每个面上的法线，然后对结果进行抗锯齿处理
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # 归一化法线图像，使得法线值范围在[0, 1]之间
                img = (img + 1) / 2
            elif type == "normal_map" :
                # 从网格的顶点属性中提取法线贴图，并对光栅化结果进行抗锯齿处理
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                # 从网格的顶点属性中提取颜色信息，并对光栅化结果进行抗锯齿处理
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)

            # 如果开启超级采样抗锯齿（SSAA > 1），则进行上采样
            if ssaa > 1:
                # 使用双线性插值将图像从超分辨率缩放到目标分辨率，`img.permute(0, 3, 1, 2)`是为了转换为(batch, channel, height, width)格式
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze() # 移除多余的维度
            else:
                # 如果没有开启SSAA，直接调整为(batch, channel, height, width)格式
                img = img.permute(0, 3, 1, 2).squeeze()
            # 将渲染结果存入字典中，键为渲染结果类型（如"mask", "color"等）
            out_dict[type] = img

        return out_dict




if __name__ == "__main__":
    perspective = torch.randn(4, 4)
    extrinsics = torch.randn(1, 4, 4)
    full_proj = (perspective @ extrinsics).unsqueeze(0)