from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline


def from_pretrained(path: str):
    """
    该函数用于加载一个模型管道。
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
        可以是本地路径或 Hugging Face 模型中心的模型名称
    """
    import os
    import json
    # 检查本地是否存在.json文件
    is_local = os.path.exists(f"{path}/pipeline.json")

    # 加载配置文件
    if is_local:
        config_file = f"{path}/pipeline.json"
    else: # 在线获取
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    # globals() 获取当前全局作用域中定义的类
    return globals()[config['name']].from_pretrained(path)
