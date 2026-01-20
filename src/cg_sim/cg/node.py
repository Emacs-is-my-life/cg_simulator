from __future__ import annotations
from enum import Enum, auto

from cg_sim.mm import Tensor


class NodeStatus(Enum):
    """TensorType represents the status of a Node"""
    TODO = auto()   # Not started yet
    SCHED = auto()  # Node is enqueued to compute unit's job queue
    RUN = auto()    # Node is being run now
    DONE = auto()   # Finished execution


class Node:
    """
    Node represents an atomic unit of workload, in a compute graph.
    Only actual computation nodes! (No GGML_OP_NONE nodes in llama.cpp)
    """

    def __init__(self, step: int, node_id: int, node_name: str,  compute_time_ns: int, output_tensor: Tensor, node_status: NodeStatus = NodeStatus.TODO):
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
        self.output_tensor = output_tensor

        """
        node_parents:  Required to check if previous jobs are finished
        node_children: For easier traversal of compute graph

        These will be initialized through Node.add_parent(Node) method.
        """
        self.parent_nodes = []   # Control Dependency
        self.children_nodes = []
        return

    def add_control_dependency(self, node_p: Node):
        """
        Adds a parent Node, for building a Compute Graph
        To fully add a (common) Node as a parent,
        you must do both:
        add_control_dependency() - Add a Node as parent node
        add_data_dependency()    - Add parent Node's output Tensor in input_tensors
        """

        # Check deuplicates
        for node in self.parent_nodes:
            if node_p.id == node.id:
                return

        # Register a parent Node
        node_p.children_nodes.append(self)
        self.parent_nodes.append(node_p)
        return

    def add_data_dependency(self, tensor_i: Tensor):
        """Adds an input Tensor, for Tensor placement check before computation"""

        # Check duplicates
        for tensor in self.input_tensors:
            if tensor_i.id == tensor.id:
                return

        # Register an input tensor
        self.input_tensors.append(tensor_i)
        return
