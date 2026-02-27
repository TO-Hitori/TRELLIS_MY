from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]):               模型字典
        sparse_structure_sampler (samplers.Sampler): 稀疏结构的采样器。
        slat_sampler (samplers.Sampler):             结构化潜在的采样器。
        slat_normalization (dict):                   结构化潜在的归一化参数。
        image_cond_model (str):                      图像条件模型的名称。
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        # 采样器设置
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        # 用于标准化的均值方差组成的字典
        self.slat_normalization = slat_normalization
        # 去除图像背景
        self.rembg_session = None
        # 图像条件模型 DINOV2
        self._init_image_cond_model(image_cond_model)


    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        # 调用父类Pipeline 的 from_pretrained 方法加载 checkpoint
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        # 创建一个空的子类实例
        new_pipeline = TrellisImageTo3DPipeline()
        # 暴力复制父类实例的所有属性到新实例。
        new_pipeline.__dict__ = pipeline.__dict__
        # 取出保存的配置字典
        args = pipeline._pretrained_args

        # 根据配置动态创建稀疏结构采样器实例：
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        初始化图像条件模型 DINOV2。
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        预处理输入图像，去除背景并调整图像大小。
        """
        # 去除背景：获取alpha
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            # 根据最大尺寸计算缩放比例
            scale = min(1, 1024 / max_size)
            # 如果缩放比例小于 1（即需要缩小图像）
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            # 创建一个新的背景去除会话
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            # 去除图像的背景，生成新的图像 output
            output = rembg.remove(input, session=self.rembg_session)
            
        # 图像裁剪和缩放
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        # 查找图像中非透明区域的坐标
        bbox = np.argwhere(alpha > 0.8 * 255)
        # 图像的边界框 [x_min, y_min, x_max, y_max]
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        # 计算边界框的中心
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        # 计算边界框的大小
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        # 新边界框
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        # 根据计算得到的新边界框 bbox 对图像进行裁剪
        # (left, upper, right, lower) -> (x_min, y_min, x_max, y_max)
        output = output.crop(bbox)  # type: ignore
        # 将图像调整为 518x518 的大小
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        # 通过alpha去除背景
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args: BCHW tensor或 PIL列表
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list): # PIL images 列表
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            # 改变尺寸518
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            # 转为 tensor
            # rgb -> 归一化 -> c通道提前 -> 堆叠为BCHW
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        # DINOv2的预处理：标准化
        image = self.image_cond_model_transform(image).to(self.device)
        # DINO特征
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        # -----features.shape torch.Size([1, 1374, 1024])
        # -----patchtokens torch.Size([1, 1374, 1024])
        
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        获取模型的条件信息: 获取条件图像和空图像的dino特征
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,            # torch.Size([1, 1374, 1024])
            'neg_cond': neg_cond,    # torch.Size([1, 1374, 1024])
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        采样并解码为稀疏结构
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        print("---------")
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        '''
        zs <class 'torch.Tensor'>
            torch.Size([1, 8, 16, 16, 16])
        decoder(z_s) <class 'torch.Tensor'>
            torch.Size([1, 1, 64, 64, 64])
            64*64*64 = 262,144
        coords <class 'torch.Tensor'>
            coords torch.Size([N, 4]) N 为非零值的数量
        '''
        
        # Decode occupancy latent
        # 解码生成的稀疏潜在表示并提取非零值的坐标。
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        
        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        解码结构化潜在表示到不同格式
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
            

        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        采样结构化潜在表示 slat。
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        ) # torch.Size([1, 8])
        
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat # torch.Size([N, 8])

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.
        运行整个管道，将图像转换为 3D 模型

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        # 预处理图像
        if preprocess_image:
            image = self.preprocess_image(image)
        # 获取DINO特征 
        cond = self.get_cond([image])  # torch.Size([1, 1374, 1024])
        # 随机种子
        torch.manual_seed(seed)
        # 采样稀疏结构 torch.Size([N, 4]) 
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        
        # 采样结构 # torch.Size([N, 8]) 
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        
        # 解码为目标格式，字典
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        注入多个图像作为条件进行采样
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.采样器名称
            num_images (int): The number of images to condition on.用于条件化的图像数量
            num_steps (int): The number of steps to run the sampler for.采样器的步骤数
        """
        # 获取指定的采样器。
        sampler = getattr(self, sampler_name)
        # 将原始的推理方法 _inference_model 保存为 _old_inference_model，以便在退出上下文时恢复。
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic': # 随机模式
            if num_images > num_steps: 
                # 如果条件图像数量大于采样步骤数，发出警告。
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion': # 多扩散模式
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    # 对不同图像进行扩散推理，并在多个条件图像之间平均预测结果
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        # 预处理图像
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        # 获取条件
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
