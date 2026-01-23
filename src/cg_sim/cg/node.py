from __future__ import annotations
from enum import Enum, auto


class NodeStatus(Enum):
    """TensorType represents the status of a Node"""
    TODO = auto()    # Not scheduled yet
    SCHED = auto()   # Node is enqueued to compute unit's job queue
    RUN = auto()     # Node is being processed now
    DONE = auto()    # Finished execution


class Node:
    """
    Node represents an atomic unit of workload, in a compute graph.
    Only actual computation nodes! (GGML_OP_NONE node of llama.cpp isn't a Node in our definition.)
    """

    def __init__(self, step: int, node_id: int, node_name: str,  compute_time_ns: int, node_status: NodeStatus = NodeStatus.TODO):
        """Initialize a Node, with it's computation characteristics"""
        self.step = step
        self.id = node_id
        self.name = node_name
        self.compute_time_ns = compute_time_ns
        self.status = node_status

        """
        tensor_input:  To check if tensors are in memory for execution of this node
        tensor_output: Same as above
        """
        self.input_tensors = []  # Data Dependency
        self.output_tensors = []

        """
        node_parents:  Required to check if previous jobs are finished
        node_children: For easier traversal of compute graph

        These will be initialized through Node.add_parent(Node) method.
        """
        self.parent_nodes = []   # Control Dependency
        self.children_nodes = []
        return

    def add_parent_node(self, node_id: int):
        """
        Adds a parent Node, for building a Compute Graph
        To fully add a (common) Node as a parent
        """

        # Check deuplicates
        for p_node_id in self.parent_nodes:
            if node_id == p_node_id:
                return

        self.parent_nodes.append(node_id)
        return

    def add_child_node(self, node_id: int):
        for c_node_id in self.children_nodes:
            if node_id == c_node_id:
                return

        self.children_nodes.append(node_id)
        return

    def add_input_tensor(self, tensor_id: int):
        """Adds an input Tensor, for Tensor placement check before computation"""

        # Check duplicates
        for i_tensor_id in self.input_tensors:
            if tensor_id == i_tensor_id:
                return

        self.input_tensors.append(tensor_id)
        return

    def add_output_tensor(self, tensor_id: int):
        # Check duplicates
        for o_tensor_id in self.output_tensors:
            if tensor_id == o_tensor_id:
                return

        self.output_tensors.append(tensor_id)
        return
