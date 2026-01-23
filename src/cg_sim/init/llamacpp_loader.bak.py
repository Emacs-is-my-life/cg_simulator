import yaml
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import read_dot
from gguf.gguf_reader import GGUFReader
import polars as pl
import re

from cg_sim.cg import NodeStatus, Node, Workload
from cg_sim.mm import TensorType, Tensor


def node_name_canonicalizer(label: str) -> str:
    """Label name from the dot graph is dirty. Get the canonical node name out of it."""
    first_part = label.split('|')[0]
    canonical_name = re.sub(r'\s\([^)]*\)$', '', first_part)
    return canonical_name


def get_tensor_type(canonical_name: str) -> TensorType:
    """Returns a TensorType, inferred from the canonical name of a tensor"""
    if ".weight" in canonical_name:
        return TensorType.WEIGHT
    elif ("cache_k" in canonical_name) or ("cache_v" in canonical_name) or ("leaf" in canonical_name):
        return TensorType.KVCache
    elif "inp_embd" == canonical_name:
        return TensorType.INPUT
    else:
        return TensorType.INTERMEDIATE


def get_tensor_by_name(node_src_name: str, TensorMap: dict[int, Tensor]) -> Tensor:
    """Using node_src_name, finds the Tensor in the TensorMap"""
    for _, tensor in TensorMap.items():
        if tensor.name == node_src_name:
            return tensor

    raise Exception("get_tensor_by_name: Could not find the source Tensor")


def get_node_output_tensor(node_record, tensor_id: int, TensorMap: dict[int, Tensor]) -> (int, Tensor):
    """
    Using node_record information, does the following:

    0. Checks if this Node's Tensor is View (virtual) or Real (new Tensor assignment)
    1. When the Tensor is View
      1.1 Finds the underlying concrete Tensor, using node_src_name
      1.2 Returns tensor_id (unmodified) and located concrete Tensor
    2. When the Tensor is Real
      2.2 Create a new Tensor
      2.2 Insert the new Tensor into the TensorMap, increment tensor_id by 1
      2.3 Returns tensor_id (++) and the new Tensor
    """

    # Check if this Node is just a view of an existing Node, or not
    if node_record["node_src_name"]:
        # View Node
        node_src_name = node_record["node_src_name"]

        return (tensor_id, get_tensor_by_name(node_src_name, TensorMap))
    else:
        # Real Node
        node_name = node_record["node_name"]

        tensor_id = tensor_id
        tensor_name = node_name
        tensor_type = get_tensor_type(node_name)
        tensor_size = node_record["node_tensor_size_bytes"]
        new_tensor = Tensor(tensor_id, tensor_name, tensor_type, tensor_size)
        TensorMap[tensor_id] = new_tensor
        tensor_id += 1

        return (tensor_id, new_tensor)


def llamacpp_loader(model_config_path: str) -> Workload:
    """
    Import llama.cpp run profiling result,
    returns a Workload object
    """

    # Read model config yaml file
    model_conf = None
    with open(model_config_path, 'r') as f:
        model_conf = yaml.safe_load(f)

    if not model_conf["type"] == "llama.cpp":
        raise Exception("[Loader] Model type mismatch!")

    model_args = model_conf["args"]
    model_path = Path(model_config_path).parent

    # Load tensor name and size from GGUF model file
    gguf_info: dict[str, int] = {} # tensor_name -> tensor_size_n_bytes
    gguf_path = model_path / model_args["gguf_path"]
    gguf_reader = GGUFReader(gguf_path)
    for tensor in gguf_reader.tensors:
        gguf_info[tensor.name] = tensor.n_bytes

    # Load node compute from profile record as a Polars DataFrame
    record_path = model_path / model_args["record_path"]
    df_records = pl.read_csv(record_path, has_header=True)

    # Load compute graph in dot format
    dot_graph_path = model_path / model_args["dot_graph_path"]
    compute_graph = read_dot(dot_graph_path)

    NodeMap: dict[int, Node] = {}
    TensorMap: dict[int, Tensor] = {}
    node_id = 0    # Increment node_id everytime new Node is added to NodeMap
    tensor_id = 0  # Increment tensor_id everytime new Tensor is added to TensorMap

    # Add Nodes & Tensors to NodeMap & TensorMap
    graph_id_map = {}
    for g_id, g_attr in compute_graph.nodes(data=True):
        node_label = g_attr["label"]

        # Check if it's a Tensor (weight node) or Node (actual computation)
        if node_label.startswith("<x>"):
            # Tensor (Data)

            canonical_name = node_name_canonicalizer(node_label[3:])  # Ditch the prefix "<x>"
            tensor_type = get_tensor_type(canonical_name)
            tensor_size_bytes = gguf_info[canonical_name]

            # Put a new Tensor into TensorMap
            new_tensor = Tensor(tensor_id, canonical_name, tensor_type, tensor_size_bytes)
            TensorMap[tensor_id] = new_tensor
            tensor_id += 1

            graph_id_map[g_id] = new_tensor
        else:
            # Node   (Computation)

            # For a Node, we have to do both:
            # 1. Add Node to NodeMap
            # 2. Add Tensor, holding the output of this Node, to TensorMap

            canonical_name = node_name_canonicalizer(node_label)
            # TODO
            #
            # Getting the first record by matching node name AND
            # step isn't 0 (not the prefill phase)
            # Will support loading many .dot files later
            node_record = df_records.filter(
                (pl.col("node_name") == canonical_name) &
                (pl.col("step") != 0)
            ).head(1).row(0, named=True)

            tensor_id, tensor = get_node_output_tensor(node_record, tensor_id, TensorMap)

            # Create a new Node
            step = node_record["step"]
            node_name = node_record["node_name"]
            compute_time_ns = node_record["compute_time_ns"]
            output_tensors = [tensor]

            # Put a new Node into NodeMap
            new_node = Node(step, node_id, node_name, compute_time_ns, output_tensors)
            NodeMap[node_id] = new_node
            node_id += 1

            graph_id_map[g_id] = new_node

    # Build a Compute Graph by adding parent-child relationships
    for parent_g_id, child_g_id, _ in compute_graph.edges():
        child_node = graph_id_map[child_g_id]
        parent = graph_id_map[parent_g_id]

        # Check if parent is Node or Tensor
        if isinstance(parent, Node):
            # Add Control Dependency
            child_node.add_control_dependency(parent)
            # Add Data Dependency
            for tensor in parent.output_tensors:
                child_node.add_data_dependency(tensor)
        elif isinstance(parent, Tensor):
            # Add Data Dependency
            child_node.add_data_dependency(parent)
        else:
            raise Exception("parent is neither Node nor Tensor")

    # Create a Workload object
    first_node_g_id = next(iter(compute_graph.nodes()))
    first_node = graph_id_map[first_node_g_id]
    ComputeGraph = first_node
    workload = Workload(ComputeGraph, NodeMap, TensorMap)

    return workload
