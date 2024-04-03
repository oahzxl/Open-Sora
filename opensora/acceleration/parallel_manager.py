import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import init_device_mesh

PARALLEL_MANAGER = None


class ParallelManager:
    def __init__(self, sp_size):
        self.sp_size = sp_size
        assert (
            dist.get_world_size() % sp_size == 0
        ), f"world size {dist.get_world_size()} must be divisible by sp_size {sp_size}"
        self.dp_size = dist.get_world_size() // self.sp_size
        self.device_mesh = init_device_mesh("cuda", (self.dp_size, self.sp_size), mesh_dim_names=("dp", "sp"))

        self.dp_group: ProcessGroup = self.device_mesh.get_group(mesh_dim="dp")
        self.dp_rank = dist.get_rank(self.dp_group)

        self.sp_group: ProcessGroup = self.device_mesh.get_group(mesh_dim="sp")
        self.sp_rank = dist.get_rank(self.sp_group)


def set_parallel_manager(sp_size):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(sp_size)


def get_data_parallel_group():
    return PARALLEL_MANAGER.dp_group


def get_data_parallel_rank():
    return PARALLEL_MANAGER.dp_rank


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.sp_rank


def get_parallel_manager():
    return PARALLEL_MANAGER
