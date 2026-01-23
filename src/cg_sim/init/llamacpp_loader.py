import yaml
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import read_dot
from gguf.gguf_reader import GGUFReader
import polars as pl
import re

from cg_sim.cg import NodeStatus, Node, TensorType, Tensor, Workload


def node_name_canonicalizer(label: str) -> str:
    """Dot graph node label -> canonical_name"""
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


def llamacpp_loader(model_config_path: str) -> Workload:
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

    # Load profiling data
    record_path = model_path / model_args["record_path"]
    df_records = pl.read_csv(record_path, has_header=True)
    df_records_step = df_records.filter(
        pl.col("step") == 6  # TODO: This is an arbitrary choice...
    )

    # Load weight metadata from the gguf file
    gguf_info = {}  # tensor_name (Weight tensors only) -> tensor_size_bytes
    gguf_path = model_path / model_args["gguf_path"]
    gguf_reader = GGUFReader(gguf_path)
    for tensor_w in gguf_reader.tensors:
        gguf_info[tensor_w.name] = tensor_w.n_bytes

    # Load dot graph
    dot_graph_path = model_path / model_args["dot_graph_path"]
    dot_graph = read_dot(dot_graph_path)

    """
    What are we doin here?

    Read three files:
      - model.gguf
      - step_x_compute_graph.dot
      - ggml_profile_node_records.csv

    then

    Build and return:
      - Workload
          - NodeMap
          - TensorMap
          - ComputeGraph (pointer to the first Node in graph)
    """

    NodeMap: dict[int, Node] = {}
    TensorWithSignMap: dict[int, TensorWithSign] = {}
    VerticeMap: dict[int, Node | TensorWithSign] = {}

    # Trackers
    node_id = 0
    tensor_id = 0

    # Iterate over all Vertices(Nodes and Tensors) in the dot graph
    for v_id, v_attr in dot_graph.nodes(data=True):
        v_label = v_attr["label"]

        # Check if it's a Tensor (Weight Tensor) or a Node (actual Computation) and Tensor (Input/Intermediate/KVCache)
        if v_label.startswith('<x>'):
            # Tensor (Weight)

            canonical_name = node_name_canonicalizer(v_label[3:])  # Ditch "<x>" part
            tensor_type = get_tensor_type(canonical_name)
            tensor_size_bytes = -1

            # <leaf> and <cache_> are assigned in runtime, so no info in gguf_info(from gguf file)
            if "leaf_" in canonical_name:
                # TODO: properly compute leaf node size, from it's v_label
                # but it's usually so tiny wouldn't matter
                tensor_size_bytes = 4  # Just 4 bytes
            elif ("cache_k" in canonical_name) or ("cache_v" in canonical_name):
                # For cache_k_lX, cache_v_lX tensors,
                # we can't locate their size in our runtime profiling data
                try:
                    cache_node_info = df_records_step.filter(
                        pl.col("node_name") == canonical_name
                    ).head(1).row(0, named=True)
                except Exception as e:
                    print(f"Cache tensor Canonical Name: {canonical_name}")
                tensor_size_bytes = cache_node_info["node_tensor_size_bytes"]
            else:
                tensor_size_bytes = gguf_info[canonical_name]

            # Put a new Tensor into TensorWithSignMap
            # Tensor signature (its runtime address) is not needed, for it is a weight Tensor.
            # There is no aliasing problem.
            tensor_new = TensorWithSign(tensor_id, canonical_name, tensor_type, "", tensor_size_bytes)
            TensorWithSignMap[tensor_id] = tensor_new
            tensor_id += 1

            VerticeMap[v_id] = tensor_new
        else:
            # Node and it's output Tensor (Input/Intermediate/KVCache)

            # For a Node, we gotta do both:
            # 1. Add a Tensor holding the output of this Node
            # 2. Add this Node

            node_info = df_records_step.filter(
                pl.col("node_n") == node_id
            ).head(1).row(0, named=True)

            canonical_name = node_name_canonicalizer(v_label)
            tensor_sign = node_info["tensor_addr"]
            tensor_real_id = get_real_tensor_id(TensorWithSignMap, tensor_sign)
            if tensor_real_id == -1:
                # Such Tensor does not exist yet in the TensorWithSignMap

                tensor_type = get_tensor_type(canonical_name)
                tensor_size_bytes = node_info["node_tensor_size_bytes"]

                # Create a new TensorWithSign and register it
                tensor_new = TensorWithSign(tensor_id, canonical_name, tensor_type, tensor_sign, tensor_size_bytes)
                TensorWithSignMap[tensor_id] = tensor_new
                tensor_id += 1

                tensor_real_id = tensor_new.id

            # Create a new Node
            step = 0
            node_name = node_info["node_name"]
            compute_time_ns = node_info["node_compute_time_ns"]
            node_new = Node(step, node_id, node_name, compute_time_ns)
            node_new.add_output_tensor(tensor_real_id)
            NodeMap[node_id] = node_new
            node_id += 1

            VerticeMap[v_id] = node_new

    # Build a Compute Graph, by translating dot graph edges into parent-child relationship
    for parent_v_id, child_v_id, _ in dot_graph.edges():
        child_node = VerticeMap[child_v_id]
        parent = VerticeMap[parent_v_id]

    # Check if the parent is Node or Tensor
    if isinstance(parent, Node):
        # Add control dependency
        parent.add_child_node(child_node.id)
        child_node.add_parent_node(parent.id)

        # Add data dependency
        for tensor_id in parent.output_tensors:
            child_node.add_input_tensor(tensor_id)
    else:
        # Add data dependency
        child_node.add_input_tensor(parent.id)

    # Create TensorMap, since we don't need tensor signatures anymore
    TensorMap: dict[int, Tensor] = {}
    for tensor_id, tensor in TensorWithSignMap.items():
        TensorMap[tensor_id] = tensor.get_tensor()

    # ComputeGraph is essentially a pointer to the start node of the whole graph
    start_node_v_id = next(iter(dot_graph.nodes()))
    start_node = VerticeMap[start_node_v_id]
    ComputeGraph = start_node

    # Create the Worload object for simulation
    workload = Workload(ComputeGraph, NodeMap, TensorMap)
    return workload
