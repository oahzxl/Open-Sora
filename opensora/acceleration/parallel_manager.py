import torch.distributed as dist
from torch.distributed import ProcessGroup

from .process_group_mesh import ProcessGroupMesh

PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, sp_size, dp_axis=0, sp_axis=1):
        super().__init__(dp_size, sp_size)
        self.dp_axis = dp_axis
        self.dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)

        self.sp_size = sp_size
        self.sp_axis = sp_axis
        self.sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)
        self.sp_rank = dist.get_rank(self.sp_group)


def set_parallel_manager(sp_size):
    global PARALLEL_MANAGER
    assert dist.get_world_size() % sp_size == 0, "world size must be divisible by sp_size"
    dp_size = dist.get_world_size() // sp_size
    PARALLEL_MANAGER = ParallelManager(dp_size, sp_size)


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
