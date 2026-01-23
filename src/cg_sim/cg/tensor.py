from enum import Enum, auto
import numpy as np


class TensorType(Enum):
    """TensorType represents the type of a Tensor"""
    INPUT = auto()
    WEIGHT = auto()
    INTERMEDIATE = auto()
    KVCache = auto()


class Tensor:
    """Tensor represents data being process in model inference"""
    def __init__(self, tensor_id: int, tensor_name: str, tensor_type: TensorType, size_bytes: int = 0, base_addr: int = -1):
        self.id = tensor_id
        self.name = tensor_name
        self.type = tensor_type
        self.size_bytes = size_bytes
        self.base_addr = base_addr

        align_bytes = 64        # 64 B in AMD64 for optimal performance
        page_size_bytes = 4096  # 4 kB in AMD64
        tensor_aligned_size_bytes = ((size_bytes + align_bytes - 1) // align_bytes) * align_bytes
        self.size_pages = int(np.ceil(tensor_aligned_size_bytes / page_size_bytes))
