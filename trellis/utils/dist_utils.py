import os
import io
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

"""
此文件包含了一些用于分布式训练的工具函数，主要实现了分布式环境的初始化、数据读取、模型解包、进程同步等功能。
这些工具函数主要用于PyTorch分布式训练中，确保多个GPU/多个节点的协调工作。

主要功能：
1. **setup_dist**：初始化分布式训练环境。设置必要的环境变量，并初始化分布式进程组，使得不同的进程可以在多个节点或多个GPU上并行训练。
2. **read_file_dist**：分布式地读取二进制文件。该函数通过`rank 0`进程读取文件内容，并将文件内容广播到所有其他进程。确保所有进程都可以访问到相同的文件数据。
3. **unwrap_dist**：解包分布式数据并行（DDP）模型。此函数从`DistributedDataParallel`（DDP）模型中提取出原始模型，因为DDP会将模型包装起来，返回的是分布式训练的版本，使用此函数可以恢复到原始模型。
4. **master_first**：上下文管理器，确保在分布式训练环境下，主进程（rank 0）先执行，而其他进程在主进程执行完毕后才能继续执行。这对于需要主进程先执行某些操作的情况非常有用。
5. **local_master_first**：上下文管理器，确保在每个本地机器上，编号为0的GPU（local master）进程先执行，而其他本地GPU进程在local master进程执行完毕后继续执行。适用于多GPU环境下，控制本地进程间的执行顺序。

这些工具函数非常适合用于大规模的分布式训练场景，其中有多个计算节点或多个GPU的情况下，确保模型和数据的正确同步，并提高训练效率。
"""



# 设置分布式训练的环境变量，初始化分布式训练进程组
def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    """
    该函数用于初始化分布式训练环境，设置进程组，并指定各个进程的设备。
    参数:
        rank (int): 当前进程的全局rank。
        local_rank (int): 当前进程在本地机器上的rank（即GPU编号）。
        world_size (int): 总的进程数量。
        master_addr (str): 主节点的IP地址。
        master_port (str): 主节点的端口号。
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    # 初始化分布式进程组，使用NCCL后端
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    

def read_file_dist(path):
    """
    分布式地读取二进制文件。
    文件仅由rank 0进程读取一次，并广播到其他进程。

    参数:
        path (str): 文件路径。
    
    返回:
        data (io.BytesIO): 读取的文件数据，以BytesIO对象的形式返回。
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        # read file
        size = torch.LongTensor(1).cuda()
        if dist.get_rank() == 0:
            with open(path, 'rb') as f:
                data = f.read()
            data = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
            ).cuda()
            size[0] = data.shape[0]
        # broadcast size
        dist.broadcast(size, src=0)
        if dist.get_rank() != 0:
            data = torch.ByteTensor(size[0].item()).cuda()
        # broadcast data
        dist.broadcast(data, src=0)
        # convert to io.BytesIO
        data = data.cpu().numpy().tobytes()
        data = io.BytesIO(data)
        return data
    else:
        with open(path, 'rb') as f:
            data = f.read()
        data = io.BytesIO(data)
        return data
    

def unwrap_dist(model):
    """
    从分布式训练中解包模型。
    如果模型是通过DistributedDataParallel包装的，则返回原始模型。

    参数:
        model (torch.nn.Module): 输入模型。
    
    返回:
        model (torch.nn.Module): 解包后的模型。
    """
    if isinstance(model, DDP):
        return model.module
    return model


@contextmanager
def master_first():
    """
    确保主进程先执行的上下文管理器。
    该管理器确保在分布式环境下主进程先执行，其他进程等待主进程执行完毕后再继续。

    使用方式：
        with master_first():
            # 主进程首先执行的代码
    """
    # 如果没有初始化分布式环境，直接执行
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() == 0: # 主进程（rank 0）
            yield                # 执行主进程的代码
            dist.barrier()       # 等待所有进程执行完毕
        else:
            dist.barrier()       # 其他进程等待主进程执行完毕
            yield                       
            

@contextmanager
def local_master_first():
    """
    确保本地主进程先执行的上下文管理器。
    该管理器确保在本地机器上，编号为0的GPU（local master）进程先执行，
    其他进程等待本地主进程执行完毕后再继续。

    使用方式：
        with local_master_first():
            # 本地主进程首先执行的代码
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    