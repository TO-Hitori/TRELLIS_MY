from abc import abstractmethod
from contextlib import contextmanager
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class MemoryController:
    """
    Base class for memory management during training.
    主要是通过记录和更新模型训练的状态以及内存使用情况来控制内存资源的分配。
    """
    
    _last_input_size = None
    _last_mem_ratio = []
    
    @contextmanager
    def record(self):
        pass
    
    def update_run_states(self, input_size=None, mem_ratio=None):
        if self._last_input_size is None:
            self._last_input_size = input_size
        elif self._last_input_size!= input_size:
            raise ValueError(f'Input size should not change for different ElasticModules.')
        self._last_mem_ratio.append(mem_ratio)
    
    @abstractmethod
    def get_mem_ratio(self, input_size):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def log(self):
        pass


class LinearMemoryController(MemoryController):
    """
    A simple controller for memory management during training.
    The memory usage is modeled as a linear function of:
        - the number of input parameters
        - the ratio of memory the model use compared to the maximum usage (with no checkpointing)
    memory_usage = k * input_size * mem_ratio + b
    The controller keeps track of the memory usage and gives the
    expected memory ratio to keep the memory usage under a target
    """
    """
    一个用于训练过程中内存管理的简单控制器。
    内存使用量被建模为以下因素的线性函数：
        - 输入参数的数量
        - 模型使用的内存与最大使用内存（不使用检查点时）的比率
    内存使用量 = k * 输入大小 * 内存比率 + b
    该控制器跟踪内存使用情况，并提供期望的内存比率，以确保内存使用不超过目标内存。
    """
    def __init__(
        self,
        buffer_size=1000,
        update_every=500,
        target_ratio=0.8,
        available_memory=None,
        max_mem_ratio_start=0.1,
        params=None,
        device=None
    ):
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.device = device or torch.cuda.current_device()
        self.available_memory = available_memory or torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                
        self._memory = np.zeros(buffer_size, dtype=np.float32)
        self._input_size = np.zeros(buffer_size, dtype=np.float32)
        self._mem_ratio = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_ptr = 0
        self._buffer_length = 0
        self._params = tuple(params) if params is not None else (0.0, 0.0)
        self._max_mem_ratio = max_mem_ratio_start
        self.step = 0

    def __repr__(self):
        return f'LinearMemoryController(target_ratio={self.target_ratio}, available_memory={self.available_memory})'
        
    def _add_sample(self, memory, input_size, mem_ratio):
        self._memory[self._buffer_ptr] = memory
        self._input_size[self._buffer_ptr] = input_size
        self._mem_ratio[self._buffer_ptr] = mem_ratio
        self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
        self._buffer_length = min(self._buffer_length + 1, self.buffer_size)
            
    @contextmanager
    def record(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self._last_input_size = None
        self._last_mem_ratio = []
        yield
        self._last_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self._last_mem_ratio = sum(self._last_mem_ratio) / len(self._last_mem_ratio)
        self._add_sample(self._last_memory, self._last_input_size, self._last_mem_ratio)
        self.step += 1
        if self.step % self.update_every == 0:
            self._max_mem_ratio = min(1.0, self._max_mem_ratio + 0.1)
            self._fit_params()
            
    def _fit_params(self):
        memory_usage = self._memory[:self._buffer_length]
        input_size = self._input_size[:self._buffer_length]
        mem_ratio = self._mem_ratio[:self._buffer_length]
        
        x = input_size * mem_ratio
        y = memory_usage
        k, b = np.polyfit(x, y, 1)
        self._params = (k, b)
        # self._visualize()
        
    def _visualize(self):
        import matplotlib.pyplot as plt
        memory_usage = self._memory[:self._buffer_length]
        input_size = self._input_size[:self._buffer_length]
        mem_ratio = self._mem_ratio[:self._buffer_length]
        k, b = self._params
        
        plt.scatter(input_size * mem_ratio, memory_usage, c=mem_ratio, cmap='viridis')
        x = np.array([0.0, 20000.0])
        plt.plot(x, k * x + b, c='r')
        plt.savefig(f'linear_memory_controller_{self.step}.png')
        plt.cla()
        
    def get_mem_ratio(self, input_size):
        k, b = self._params
        if k == 0: return np.random.rand() * self._max_mem_ratio
        pred = (self.available_memory * self.target_ratio - b) / (k * input_size)
        return min(self._max_mem_ratio, max(0.0, pred))
    
    def state_dict(self):
        return {
            'params': self._params,
        }
        
    def load_state_dict(self, state_dict):
        self._params = tuple(state_dict['params'])
        
    def log(self):
        return {
            'params/k': self._params[0],
            'params/b': self._params[1],
            'memory': self._last_memory,
            'input_size': self._last_input_size,
            'mem_ratio': self._last_mem_ratio,
        }
    
    
class ElasticModule(nn.Module):
    """
    Module for training with elastic memory management.
    在训练过程中实现 弹性内存管理。弹性内存管理的核心思想是根据模型的输入大小动态调整内存使用比例，从而平衡内存消耗与计算开销
    """
    def __init__(self):
        super().__init__()
        self._memory_controller: MemoryController = None
        
    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        """
        Get the size of the input data.
        
        Returns:
            int: The size of the input data.
        """
        pass
    
    @abstractmethod
    def _forward_with_mem_ratio(self, *args, mem_ratio=0.0, **kwargs) -> Tuple[float, Tuple]:
        """
        Forward with a given memory ratio.
        """
        pass
    
    def register_memory_controller(self, memory_controller: MemoryController):
        self._memory_controller = memory_controller
        
    def forward(self, *args, **kwargs):
        if self._memory_controller is None or not torch.is_grad_enabled() or not self.training:
            _, ret = self._forward_with_mem_ratio(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            mem_ratio, ret = self._forward_with_mem_ratio(*args, mem_ratio=mem_ratio, **kwargs)
            self._memory_controller.update_run_states(input_size, mem_ratio)
        return ret
    

class ElasticModuleMixin:
    """
    Mixin for training with elastic memory management.
    用于支持在训练过程中进行 弹性内存管理。ElasticModuleMixin 提供了内存管理的框架，
    并结合上下文管理器和 MemoryController 类来动态调整训练过程中的内存使用比例（mem_ratio）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_controller: MemoryController = None
        
    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        """
        Get the size of the input data.
        
        Returns:
            int: The size of the input data.
        """
        pass
    
    @abstractmethod
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0) -> float:
        """
        Context manager for training with a reduced memory ratio compared to the full memory usage.
        
        Returns:
            float: The exact memory ratio used during the forward pass.
        """
        pass
    
    def register_memory_controller(self, memory_controller: MemoryController):
        self._memory_controller = memory_controller
        
    def forward(self, *args, **kwargs):
        if self._memory_controller is None or not torch.is_grad_enabled() or not self.training:
            ret = super().forward(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                ret = super().forward(*args, **kwargs)
            self._memory_controller.update_run_states(input_size, exact_mem_ratio)
        return ret
