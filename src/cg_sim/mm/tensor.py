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
    def __init__(self, tensor_id: int, tensor_type: TensorType, base_addr: int, size_bytes: int, annotation: str):
        self.tensor_id = tensor_id
        """Unique ID to identifier a Tensor"""
        self.tensor_type = tensor_type
        """Type of the Tensor"""
        self.base_addr = base_addr
        """
        Memory address where the first byte of Tensor resides.
        base_addr is -1 if Tensor is offloaded to disk.
        """
        self.annotation = annotation
        """
        Extra information about Tensor.
        For example, Tensor name, etc
        """

        align_bytes = 64       # 64 B in AMD64 for optimal performance
        page_size_bytes = 4096 # 4 kB in AMD64
        tensor_aligned_size_bytes = ((size_bytes + align_bytes - 1) // align_bytes) * align_bytes
        self.size_pages = int(np.ceil(tensor_aligned_size_bytes / page_size_bytes))
        """
        Tensor size, represented in number of pages.
        Anticipated alignment in memory (64 B cache line alignment)
        So Tensor size in bytes would be size_pages * 4096.
        """
