from .node import Node
from .tensor import Tensor


def Map_Check(NodeMap: dict[int, Node], TensorMap: dict[int, Tensor]) -> bool:
    """Check if data dependency (tensor) are well-defined"""
    for node in NodeMap.values():
        # Check if input tensors are all defined in the TensorMap
        for tensor_i in node.input_tensors:
            if not TensorMap.get(tensor_i.id):
                raise Exception(f"Node {node.id}'s input tensor: Tensor {tensor_i.id} does not exist in the TensorMap.")

        # Check if output tensors are all defined in the TensorMap
        for tensor_o in node.output_tensors:
            if not TensorMap.get(tensor_o.id):
                raise Exception(f"Node {node.id}'s output tensor: Tensor {tensor_o.id} does not exist in the TensorMap.")

    return True


def ComputeGraph_Check(ComputeGraph: Node, NodeMap: dict[int, Node]) -> bool:
    """Check if provided ComputeGraph is forming a proper Directed Acylic Graph"""

    # TODO
    return True


class Workload:
    """LLM inference workload to process in digestible format for the scheduler"""

    def __init__(self, ComputeGraph: Node, NodeMap: dict[int, Node], TensorMap: dict[int, Tensor]):

        Map_Check(NodeMap, TensorMap)
        ComputeGraph_Check(ComputeGraph, NodeMap)

        self.ComputeGraph = ComputeGraph  # Root Node
        self.NodeMap = NodeMap            # Map of Nodes (id -> Node)
        self.TensorMap = TensorMap        # Map of Tensors (id -> Tensor)
        return
