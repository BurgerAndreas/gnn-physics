

def split_edge(edges, nodes, edge_to_split):
    """Adds a node
    If we add a node:
    - we remove the edge_to_split
    - we add four edges
        - two edges along were the edge_to_split was
        - two edges to neighbouring nodes
    """
    return edges

def collapse_edge(edges, nodes, edge_to_collapse):
    """Removes a node
    If we remove a node from a triangular mesh:
    - we remove four edges (three if node is at the boundary)
    - 
    - we get a quad
    We can either (a) think of removing edges (b) think of removing nodes
    Advantage of removing edges: obvious how to split the quad
    Disadvante of removing edges: need to check the removal of two nodes
    """
    # if edge is at the boundary, do nothing

    # decide which way to split the quad 
    # = between which nodes to add a new edge.
    # the removed node was connected to four other nodes:
    # the other node_i of edge_to_collapse
    # two nodes which are also connected to node_i
    # node_k which is not connected to node_i yet
    # -> add the new edge between node_i and node_k (edge_ik)
    return edges, nodes


def local_remesher():
    """
    three fundamental operations: splitting an edge to refine the mesh
    collapsing an edge to coarsen it, 
    and flipping an edge to change orientation
    in: 
    sizing field tensor S_i at each node n_i
    nodes n_i and edges
    """
    edges = {edge_ij: (node_i, node_j)}
    nodes = {node_i: (edge_ij, edge_ik, edge_il, edge_im)}
    seizing_tensors = {node_i: s_i}

    # 1. split edges
    for edge in edges:
        u_ij = ???
        avg_sizing_tensor = (s_i + s_j) / 2
        if u_ij.T @ avg_sizing_tensor @ u_ij > 1:
            edges = split_edge(edges, edge_ij)
    
    # 2. flip edges the first time
    for edge_ij in edges:
        # find the other possible edge_kl
        # between the two nodes (k,l) connected to both node_i and node_j
        node_k_node_l = []
        for edge_ik in nodes[i]:
            for edge_jk in nodes[j]:
                if edge_ik == edge_jk:
                    node_k_node_l.append(node_k)
                    break
        # test the an-isotropic Delaunay criterion
        # A pliant method for anisotropic mesh generation, 1996
    
    # 3. collapse edges
    for edge in edges:
        # try collapsing
        # if no new edge is invalid, perform collapse
        edges_collapsed = edges.copy()
        for edge in nodes[1]:
            edges_collapsed = collapse_edge(edges_collapsed, edge)
    
    # 4. flip edges a second time


