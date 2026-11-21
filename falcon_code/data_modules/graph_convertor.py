import torch
import networkx as nx
from torch_geometric.data import Data


def networkx_to_pyg(nx_graph):
    """
    Converts a NetworkX MultiGraph with all attributes (numeric and string) into a PyTorch Geometric Data object.

    Args:
        nx_graph (networkx.MultiGraph): The NetworkX graph to be converted.

    Returns:
        tuple: (torch_geometric.data.Data, dict, dict)
               - PyG Data object
               - Node mapping (original names → indices)
               - Dictionary of all attributes (numeric & string)
    """
    # Map node names to indices
    # nx_graph.nodes()：获取所有已经存在的节点
    # enumerate(nx_graph.nodes()) 给每个节点编号
    # node_mapping = {
    #     "Vin": 0,
    #     "net1": 1,
    #     "GND": 2,
    #     "Vout": 3
    # }
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}

    # Create node feature tensor (Modify if real features exist)
    # [[0], [1], [2], [3], ..., [n]]
    # tensor就是适合机器学习的多维数组
    x = torch.tensor([[i] for i in range(len(nx_graph.nodes()))], dtype=torch.float)

    edge_index = []
    edge_attr_dict = {}  # Dictionary to store all attributes

    # u是edge连接的第一个节点
    # v是edge连接的第二个节点
    # key是用来区分同意对节点之间不同边的编号, 由 NetworkX 的 MultiGraph 自动分配的
    # data就是这条边带有的所有属性, name. component, attr
    for u, v, key, data in nx_graph.edges(keys=True, data=True):
        # 之前node_mapping已经让node与数字联系起来, 现在则是把原先node的描述方式用对应数字表达
        # edge_index = [
        #     [0, 1],
        #     [0, 1],
        #     [1, 2]
        # ]
        edge_index.append([node_mapping[u], node_mapping[v]])  # Convert nodes to indices

        # Store all attributes (numeric and string) together
        # edge_attr_dict = {
        #     (0, 1, 0): {
        #         "name": "R1",
        #         "component": "resistor",
        #         "r": 100
        #     },
        #
        #     (0, 1, 1): {
        #         "name": "R2",
        #         "component": "resistor",
        #         "r": 200
        #     }
        # }
        edge_attr_dict[(node_mapping[u], node_mapping[v], key)] = data.copy()

    # Convert edge_index to tensor
    # 通过.t()专制后再变为tensor, 相当于两行分别保存每条边的两个端点，而不是每行保存一条边
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)  # No separate edge_attr

    return data, node_mapping, edge_attr_dict  # Return all attributes together


def pyg_to_networkx(pyg_data, node_mapping, edge_attr_dict):
    """
    Converts a PyTorch Geometric Data object back to a NetworkX MultiGraph, 
    restoring both numeric and string attributes.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object.
        node_mapping (dict): Mapping from original NetworkX node names to PyG indices.
        edge_attr_dict (dict): Stored attributes (numeric & string).

    Returns:
        networkx.MultiGraph: The reconstructed NetworkX MultiGraph.
    """
    # Create an empty MultiGraph
    nx_graph = nx.MultiGraph()

    # Reverse the node mapping (PyG index -> original node name)
    reverse_mapping = {v: k for k, v in node_mapping.items()}

    # Add nodes back
    for i in range(pyg_data.x.shape[0]):
        nx_graph.add_node(reverse_mapping[i])

    # Convert edges back
    edge_index = pyg_data.edge_index.t().tolist()  # Convert tensor to list of edges

    # Reconstruct edges with attributes
    for (u, v) in edge_index:
        key = len(nx_graph.get_edge_data(reverse_mapping[u], reverse_mapping[v], default={}))  # Track multi-edges

        # Restore all attributes (numeric & string)
        attr_dict = edge_attr_dict.get((u, v, key), {})
        nx_graph.add_edge(reverse_mapping[u], reverse_mapping[v], **attr_dict)

    return nx_graph

