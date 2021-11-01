import tensorflow as tf
import numpy as np


def get_sender_receiver(node_features, edge_index):
    '''
    Collects features from sending nodes and corresponding receiving
    nodes and returns both, e.g.
    A -> B
    A -> C
    C -> B
    returns the sender features [A, A, C] 
    and receiver features [B, C, B]
    '''
    senders = tf.gather(node_features, edge_index[:,0])
    receivers = tf.gather(node_features, edge_index[:,1])
    return senders, receivers

def get_in_degrees(edge_index, expand_edges=True, n_nodes=None):
    '''
    Returns the in degrees of the nodes, i.e. 
    how many incoming edges every node has.
    if expand_edges is True, the node degrees are
    gathered according to edges. If the edges are
    A -> B
    A -> C
    C -> B
    returns [deg(B), deg(C), deg(B)]
    '''
    uniq = tf.unique_with_counts(edge_index[:,1])
    in_degrees = uniq.count
    if n_nodes is not None:
        in_degrees = tf.scatter_nd(uniq.y[:,tf.newaxis], in_degrees, [n_nodes])
    if expand_edges:
        in_degrees = tf.gather(in_degrees, uniq.idx)
    return in_degrees

def get_out_degrees(edge_index, expand_edges=True, n_nodes=None):
    '''
    Returns the out degrees of the nodes, i.e. 
    how many outgoing edges every node has. 
    if expand_edges is True, the node degrees are
    gathered according to edges. If the edges are
    A -> B
    A -> C
    C -> B
    returns [deg(A), deg(A), deg(C)]
    '''
    uniq = tf.unique_with_counts(edge_index[:,0])
    out_degrees = uniq.count
    if n_nodes is not None:
        out_degrees = tf.scatter_nd(uniq.y[:,tf.newaxis], out_degrees, [n_nodes])
    if expand_edges:
        out_degrees = tf.gather(out_degrees, uniq.idx)
    return out_degrees

def segment_softmax(x, segment_ids, num_segments):
    '''
    Computes the softmax over segments of x defined by segment_ids.
    Can be used to apply softmax to a neighborhood in a graph (e.g. for GAT).
    Example:
    segments = [1,     0,     0,     1,     1   ]
    data     = [3.,    1.,    1.,    3.,    3.  ]
    output   = [0.33,  0.5,   0.5,   0.33,  0.33]
    '''
    x = tf.exp(x)
    x_sum = tf.math.unsorted_segment_sum(x, segment_ids, num_segments=num_segments)
    x_sum = tf.gather(x_sum, segment_ids)
    return x / x_sum

def index2adj(edge_index, n_nodes=None, dtype=tf.uint8):
    '''
    Converts edge_index representation into an adjacency matrix.
    '''
    n_nodes = tf.reduce_max(edge_index) + 1 if n_nodes is None else n_nodes
    n_edges = edge_index.shape[0]
    values = tf.ones(n_edges, dtype=dtype)
    adj = tf.scatter_nd(edge_index, values, [n_nodes, n_nodes])
    return adj

def adj2index(adj, dtype=tf.int32):
    '''
    Converts an adjacency matrix to edge_index representation.
    '''
    return tf.cast(tf.where(adj), dtype=dtype)

def get_khop_index(edge_index, k=2, n_nodes=None, output_intermediate_graphs=False):
    '''
    Computes the k-hop graph from the 1-hop graph given
    by edge_index.
    '''
    one_hop = index2adj(edge_index, n_nodes, tf.int32)

    if output_intermediate_graphs:
        k_hop = tf.scan(lambda x, y: adj2index(index2adj(x, n_nodes, tf.int32) @ one_hop), 
            tf.range(k-1), initializer=edge_index)
        k_hop = tf.concat((edge_index[tf.newaxis], k_hop), axis=0)
    else:
        k_hop = index2adj(edge_index, n_nodes=n_nodes)

        for i in range(k-1):
            k_hop = k_hop @ one_hop
        
        k_hop = adj2index(k_hop)

    return k_hop

def get_khop_index_np(edge_index, factors, n_nodes=None):
    max_factor = max(factors)
    n_nodes = tf.reduce_max(edge_index) + 1 if n_nodes is None else n_nodes
    
    def index2adj(edges, n_nodes): 
        adj = np.zeros([n_nodes, n_nodes], dtype=np.int8)
        idx = edges.T
        adj[idx[0], idx[1]] = 1
        return adj
    
    def adj2index(adj):
        return np.stack(np.where(adj), axis=-1)
    
    adj = index2adj(edge_index, n_nodes=n_nodes)
    diffusions = []
    edges = edge_index
    
    for d in range(2, max_factor+1):
        edges = adj2index(index2adj(edges, n_nodes) @ adj)
        print(d)
        if d in factors:
            diffusions.append(edges)
        print(d)
    
    return diffusions

def diffuse(e_index, n_nodes, input_edges=None):
    diffuse_with = lambda g1, g2: _diffuse(g1, n_nodes, g2)
    pow_diffuse = lambda g: _diffuse(g, n_nodes)

    if tf.nest.is_nested(e_index):
        if input_edges is not None:
            e_index = tf.nest.map_structure(diffuse_with, e_index, input_edges)
        else:
            e_index = tf.nest.map_structure(pow_diffuse, e_index)
    else:
        e_index = diffuse(e_index, n_nodes, input_edges)
    return e_index

def _diffuse(e_index, n_nodes, input_edges=None):
    '''
    Diffueses the graph with itself if input_edges is None, i.e. 
    consecutive diffusions happen in powers of two if input_edges is not given.
    '''
    rg_ns = tf.ragged.stack_dynamic_partitions(e_index[:,1], e_index[:,0], n_nodes)
    if tf.is_tensor(input_edges):
        ie_ns = tf.ragged.stack_dynamic_partitions(input_edges[:,1], input_edges[:,0], n_nodes)
        new_neighbors = tf.gather(ie_ns, rg_ns)
    else:
        new_neighbors = tf.gather(rg_ns, rg_ns)
    new_neighbors = new_neighbors.merge_dims(-2, -1)
    new_rec = new_neighbors.flat_values
    new_snd = tf.repeat(
        tf.range(n_nodes), new_neighbors.row_lengths(), axis=0)
    return tf.stack((new_snd, new_rec), axis=-1)
