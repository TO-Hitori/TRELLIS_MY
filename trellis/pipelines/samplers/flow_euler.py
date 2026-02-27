from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
                    噪声的最小尺度
    x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps
    
    v = - x_0 + (1 - self.sigma_min) * eps
    """
    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    # 从噪声 eps 反推开始的图像 x_0，用于生成模型采样时的中间步骤。
    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    # 从开始的图像 x_0 推导出噪声 eps，用于生成模型采样时的中间步骤。
    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    # 基于速度 获取 x_0 和 eps
    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        # 噪声 = 当前时刻 + 剩余时间 x 速度
        eps = x_t + (1 - t) * v
        # 噪声调整
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    # 模型预测出 速度v
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        '''
        model: 用于生成预测的模型。
        x_t:   当前时间步的输入图像（噪声图像）。
        t:     当前时间步。
        cond:  条件信息（可选）。
        '''
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            # 扩展条件批量
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        # 模型进行推理。
        return model(x_t, t, cond, **kwargs)

    # 模型预测出 速度v, x_0, eps
    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        # 模型预测当前时间 t 对应的速度 v
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        # 通过速度 v 计算当前时间 t 对应的 x_0 和 eps
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    # 一步采样
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        使用 Euler 方法从模型中生成一个采样的图像 x_{t-1}
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        # 预测 v, x_0, eps
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        # 根据时间间隔计算上一步
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.   
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # 初始噪声，采样的起点
        sample = noise 
        # 时间序列，根据步数等间隔划分
        t_seq = np.linspace(1, 0, steps + 1) 
        # rescale_t = 1，相当于不变
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        # 构建相邻的时间对 [t, t-1]
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        # 构建字典，保存采样值
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # 遍历时间对
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            # 一次采样
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            # 跟新采样值 x_{t-1}
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        通过ClassifierFreeGuidanceSamplerMixin替换推理流程，增加CFG功能
        
        neg_cond 和 cfg_strength 是新增的参数，用于实现无分类器引导的功能。
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    追加时间区间的CFG功能
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
