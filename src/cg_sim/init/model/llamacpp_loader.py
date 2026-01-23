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
    Tensor data type, but with its runtime address as signature
    Signature is required to distinguish real Tensor and virtual Tensor from traces
    """
    def __init__(self, tensor_id: int, tensor_name: str, tensor_type: TensorType, tensor_sign: str, size_bytes: int = 0, base_addr: int = -1):
        super().__init__(tensor_id, tensor_name, tensor_type, size_bytes, base_addr)
        self.sign = tensor_sign
        return

    def get_Tensor(self):
        return Tensor(self.id, self.name, self.type, self.size_bytes, self.base_addr)


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

    # Load profiling data
    record_path = model_path / model_args["record_path"]
    df_records = pl.read_csv(record_path, has_header=True)
    df_records_step = df_records.filter(
        pl.col("step") == 6  # TODO: This is an arbitrary choice...
    )

    # Load weight metadata from the gguf file
    gguf_info = dict[str, int] = {}  # tensor_name (Weight tensors only) -> tensor_size_bytes
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
    GraphIDMap: dict[int, Node | TensorWithSign] = {}

    # Trackers
    node_id = 0
    tensor_id = 0

    # Iterate over all Vertices(Nodes and Tensors) in the dot graph
    for g_id, g_attr in dot_graph.nodes(data=True):
        v_label = g_attr["label"]

        # Check if it's a Tensor (Weight Tensor) or a Node (actual Computation) and Tensor (Input/Intermediate/KVCache)
        if v_label.startswith('<x>'):
            # Tensor (Weight)

            canonical_name = node_name_canonicalizer(v_label[3:])  # Ditch "<x>" part
            tensor_type = get_tensor_type(canonical_name)
            tensor_size_bytes = gguf_info[canonical_name]

            # Put a new Tensor into TensorWithSignMap
            # Tensor signature (its runtime address) is not needed, for it is a weight Tensor.
            # There is no aliasing problem.
            tensor_new = TensorWithSign(tensor_id, canonical_name, tensor_type, "", tensor_size_bytes)
            TensorWithSignMap[tensor_id] = tensor_new
            tensor_id += 1

            GraphIDMap[g_id] = tensor_new
        else:
            # Node and it's output Tensor (Input/Intermediate/KVCache)

            # For a Node, we gotta do both:
            # 1. Add a Tensor holding the output of this Node
            # 2. Add this Node

            node_info = df_records_step.filter(
                pl.col("node_n") == node_id
            ).head(1).row(0, named=True)

            





    # return workload
