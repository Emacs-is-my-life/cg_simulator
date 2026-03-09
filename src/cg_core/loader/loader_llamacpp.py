import yaml
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import read_dot
import polars as pl
import re

from cg_sim.cg import NodeStatus, Node, TensorType, Tensor, Workload


def node_name_canonicalizer(label: str) -> str:
    """Dot graph node label -> canonical_name"""
    first_part = label.split('|')[0]
    canonical_name = re.sub(r'\s\([^)]*\)$', '', first_part)
    return canonical_name


def get_tensor_type(label: str) -> TensorType:
    """Returns a TensorType, inferred from the label name of the node"""
    if label.startswith("<x>"):
        if ".weight" in label:
            return TensorType.WEIGHT
        elif ("cache_k" in label) or ("cache_v" in label) or ("leaf" in label):
            return TensorType.KVCache
        elif "inp_embd" == label:
            return TensorType.INPUT
    else:
        return TensorType.INTERMEDIATE


class TensorWithSign(Tensor):
    """
    Tensor data type, but with its runtime address as a signature
    Signature is required to distinguish real Tensor and virtual Tensor from traces
    """
    def __init__(self, tensor_id: int, tensor_name: str, tensor_type: TensorType, tensor_sign: str, size_bytes: int = 0, base_addr: int = -1):
        super().__init__(tensor_id, tensor_name, tensor_type, size_bytes, base_addr)
        self.sign = tensor_sign
        return

    def get_Tensor(self):
        return Tensor(self.id, self.name, self.type, self.size_bytes, self.base_addr)


def get_real_tensor_id(TensorWithSignMap: dict[int, TensorWithSign], tensor_sign: str) -> int:
    """
    Using tensor_sign, look for the existing Tensor in TensorWithSignMap.
    If there is no such Tensor, returns -1.
    """

    for tensor_id, tensor in TensorWithSignMap.items():
        if tensor_sign == tensor.sign:
            return tensor_id

    return -1


def TraceLoader(model_config_path: str) -> Workload:
    """
    Import llama.cpp run profiling result,
    returns a Workload object
    """

    # Read model config yaml file
    model_conf = None
    with open(model_config_path, 'r') as f:
        model_conf = yaml.safe_load(f)["model"]

    if not model_conf["type"] == "llama.cpp":
        raise Exception("[Loader] Model type mismatch!")

    model_args = model_conf["args"]
    model_path = Path(model_config_path).parent

    # Load the dot graph
    dot_graph_path = model_path / model_args["dot_graph_path"]
    dot_graph = read_dot(dot_graph_path) 

    # Load profiled record data
    record_path = model_path / model_args["record_path"]
    df_profile_records = pl.read_csv(record_path, has_header=True)

    # Data structures for tracking Nodes and Tensors
    vNodeMap: dict[int, Node] = {}
    vTensorMap: dict[int, TensorWithSign] = {}
    VerticeMap: dict[int, Node | TensorWithSign] = {}

    # ID tracker. Increment by 1 everytime adding a Node/Tensor
    v_node_id = 0
    v_tensor_id = 0

    # Iterate over all Vertices(Nodes + Tensors) in the dot graph
    for v_id, v_attr in dot_graph.nodes(data=True):
        v_label = v_attr["label"]
        tensor_type = get_tensor_type(v_label)
        tensor_sign = v_attr["addr"]
        tensor_size_bytes = int(v_attr["size"])

        # This vertice is only a tensor, not a computation node (= leaf)
        if v_label.startswith('<x>'):
            tensor_name = node_name_canonicalizer(v_label[3:])  # Ditch "<x>" part

            __tensor_id = get_real_tensor_id(vTensorMap, tensor_sign)
            if __tensor_id == -1:
                new_tensor = TensorWithSign(v_tensor_id, tensor_name, tensor_type, tensor_sign, tensor_size_bytes)
                vTensorMap[v_tensor_id] = new_tensor
                v_tensor_id += 1

                VerticeMap[v_id] = new_tensor
            else: 
                VerticeMap[v_id] = vTensorMap[__tensor_id]

        # This vertice is a node, actual computation 
        else:
            tensor_name = node_name_canonicalizer(v_label)

            __tensor_id = get_real_tensor_id(vTensorMap, tensor_sign)
            if __tensor_id == -1:
                new_tensor = TensorWithSign(v_tensor_id, tensor_name, tensor_type, tensor_sign, tensor_size_bytes)
                vTensorMap[v_tensor_id] = new_tensor
                __tensor_id = v_tensor_id
                v_tensor_id += 1

            # Create a new Node
            step = -1  # Virtual Node
            node_name = tensor_name
            compute_time_ns = -1  # Virtual Node
            new_node = Node(step, v_node_id, node_name, compute_time_ns)
            new_node.add_output_tensor(__tensor_id)
            vNodeMap[v_node_id] = new_node
            v_node_id += 1

            VerticeMap[v_id] = new_node

    # Iterate over edges in the dot graph, to install dependency between:
    # Node <-> Node    (Control Dependency)
    # Node <-> Tensor  (Data Dependency)
    for parent_v_id, child_v_id in dot_graph.edges():
        child = VerticeMap[child_v_id]
        parent = VerticeMap[parent_v_id]

        # Parent is a Node (Control Dependency)
        if isinstance(parent, Node):
            # Add Control Dependency
            parent.add_child_node(child.id)
            child.add_parent_node(parent.id)

            # Add Data Dependency
            for tensor_id in parent.output_tensors:
                child.add_input_tensor(tensor_id)

        # Parent is a Tensor (Leaf, Data Dependency)
        else:
            # Add Data Dependency
            child.add_input_tensor(parent.id)

    # Prepare real NodeMap and TensorMap
    NodeMap: dict[int, Node] = {}
    TensorMap: dict[int, Tensor] = {}

    # vTensorMap -> TensorMap
    for tensor in vTensorMap.values():
        real_tensor = Tensor(tensor.id, tensor.name, tensor.type, tensor.size_bytes, tensor.base_addr)
        TensorMap[real_tensor.id] = real_tensor

    # vNodeMap + df_profile_records -> NodeMap
    node_id = 0
    step_till = 10
    for step in range(step_till+1):
        df_step = df_profile_records.filter(
            pl.col("step") == step
        )

        step_node_id_base = node_id  # Starting node_id of this step

        for _node_data in df_step.iter_rows():
            _node_id = _node_data[1]
            _node_name = _node_data[2]
            _node_compute_time_ns = _node_data[4]

            new_node = Node(step, node_id, _node_name, _node_compute_time_ns)
            if (step > 0) and (_node_id == 0):
                # Link to the existing compute graph.
                last_node = NodeMap[node_id - 1]

                # Add Control Dependency
                new_node.add_parent_node(last_node.id)
                last_node.add_child_node(new_node.id)

                # Add Data Dependency
                for tid in last_node.output_tensors:
                    new_node.add_input_tensor(tensor_id)

            # Copy traits from the vNodeMap
            virtual_node = vNodeMap[_node_id]
            for nid in virtual_node.parent_nodes:
                new_node.add_parent_node(step_node_id_base + nid)

            for nid in virtual_node.children_nodes:
                new_node.add_child_node(step_node_id_base + nid)

            for tid in virtual_node.input_tensors:
                new_node.add_input_tensor(tid)

            for tid in virtual_node.output_tensors:
                new_node.add_output_tensor(tid)

            # Add new_node to NodeMap
            NodeMap[node_id] = new_node
            node_id += 1

    return Workload(NodeMap, TensorMap)
