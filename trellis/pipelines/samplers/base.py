from typing import *
from abc import ABC, abstractmethod



class Sampler(ABC):
    """
    A base class for samplers.
    抽象类，不能直接实例化。
    它旨在作为其他类的蓝图，这些类会实现不同的采样方法。    
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.
        """
        pass
    