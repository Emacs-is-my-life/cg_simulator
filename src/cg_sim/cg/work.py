from .node import Node
from .tensor import Tensor


def Map_Check(NodeMap: dict[int, Node], TensorMap: dict[int, Tensor]):
    """
    Check if there is no missing Node or Tensor in Maps
    """

    for node in NodeMap.values():
        # Check if input tensors are all defined in the TensorMap
        for tensor_id in node.input_tensors:
            if not TensorMap.get(tensor_id):
                raise Exception(f"Node {node.id}'s input tensor: Tensor {tensor_id} does not exist in the TensorMap.")

        # Check if output tensors are all defined in the TensorMap
        for tensor_id in node.output_tensors:
            if not TensorMap.get(tensor_id):
                raise Exception(f"Node {node.id}'s output tensor: Tensor {tensor_id} does not exist in the TensorMap.")

    return


class Workload:
    """
    Datastructure representing LLM inference workload to process.
    """

    def __init__(self, NodeMap: dict[int, Node], TensorMap: dict[int, Tensor]):
        Map_Check(NodeMap, TensorMap)

        """
        First Node in the NodeMap, should be the initial node in the compute graph!
        """
        self.NodeMap = NodeMap            # Map of Nodes (id -> Node)
        self.TensorMap = TensorMap        # Map of Tensors (id -> Tensor)
        return
