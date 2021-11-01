import layers.graph_layers as graph_layers
import layers.graph_ops as graph_ops
import tensorflow as tf
import layers as L
import tensorflow.keras.layers as layers
import numpy as np

DEBUGGING = False

class GeoQuadrantGCN(tf.keras.layers.Layer):
    '''
    Apply separate GCNs to 4 subgraphs and sum up their results.
    The graph of the pixel grid is converted into 4 subgraphs that contain
    only north-, east-, south-, and westbound connections respectively.
    Thereby the layer can discriminate the neighbors of a pixel by global direction.
    This should lead to a better optimization, as directional information is inherent 
    in the data, but GNNs cannot utilize this information, whereas this layer can.
    '''
    
    def __init__(self, units, activation=None, use_bias=False, combination='concat', 
        learn_global_scale=True, use_global=False, **kwargs):
        super(GeoQuadrantGCN, self).__init__()
        self.units = units
        self.activation_fn = activation
        self.activation = tf.keras.layers.Activation(activation)
        self.use_bias = use_bias
        self.gcns = []
        self.global_scale = 1.
        self.global_scale_nodes = 1.
        self.global_scale_edges = 1.
        self.use_global = use_global
        self.learn_global_scale = learn_global_scale
        
        if combination == 'concat':
            units = units // 4
            self.call = self.call_concat
        if combination == 'additive':
            self.call = self.call_additive
        
        self.node_embeddings = []
        self.edge_transforms = []
        self.node_transforms = []
        self.glob_transforms = []
        self.group_norms = []
        
        for i in range(4):
            self.node_embeddings.append(tf.keras.layers.Dense(units, activation=None, use_bias=self.use_bias))
            self.edge_transforms.append(tf.keras.layers.Dense(units, activation=None, use_bias=self.use_bias))
            #self.node_transforms.append(tf.keras.layers.Dense(units, activation=None))
            if self.use_global:
                self.glob_transforms.append(tf.keras.layers.Dense(units, activation=None, use_bias=self.use_bias))
            else:
                self.glob_transforms.append(lambda x: x)
            #self.group_norms.append(tfa.layers.GroupNormalization(groups=units//2))
        self.node_transform = tf.keras.layers.Dense(self.units, activation=None, use_bias=self.use_bias)
        
        #self.group_norm = tfa.layers.GroupNormalization(groups=units//2)
        #self.edge_transform = tf.keras.layers.Dense(self.units, activation=activation)
            
        self.max_neighbors = 3
        self.scale = tf.constant(1./(self.max_neighbors+1), dtype=tf.float32)
        #self.eps = tf.Variable(tf.zeros([1]), trainable=True, name='GIN_eps')
    
    def build(self, input_shape):
        if self.use_global and self.learn_global_scale:
            self.glob_transform = tf.keras.layers.Dense(self.units, activation=self.activation_fn, use_bias=self.use_bias)
            self.global_scale_nodes = tf.Variable(self.global_scale, trainable=True)
            self.global_scale_edges = tf.Variable(self.global_scale, trainable=True)
        else:
            self.glob_transform = tf.identity

    def call_concat(self, x, graph, training=False):
        n_nodes = tf.shape(x)[0]
        edge_index = graph['edge_index']
        edges = graph['edges']
        n_edge_types = len(edge_index)

        edge_global = 0.
        node_global = 0.
        quad_nodes = []
        quad_neighbors = []
        quad_edges = []
        glob = graph['global']

        for linear1, linear2, linear3, e_feat, e_index in zip(self.node_embeddings, 
                                                      self.edge_transforms, 
                                                      self.glob_transforms,
                                                      edges,
                                                      edge_index):
            # project node features to a lower dimensional space
            x_quad = linear1(x)
            x_quad = self.activation(x_quad)
            sender, receiver = graph_ops.get_sender_receiver(x_quad, e_index)

            edge = tf.concat((e_feat, sender, receiver), axis=-1)

            if self.use_global:
                glob = linear3(glob)
                glob = self.activation(glob)
                _global = tf.tile(glob, [tf.shape(edge)[0], 1])
                edge = tf.concat((edge, _global), axis=-1)

            edge = linear2(edge)
            edge = self.activation(edge)
            
            if self.use_global:
                # this is problematic, here directional information is lost
                edge_global += tf.reduce_sum(edge, axis=0, keepdims=True)

            neighbor_agg = tf.math.unsorted_segment_sum(edge, e_index[:,1], 
                                                        num_segments=n_nodes)

            neighbor_agg *= self.scale
            quad_neighbors.append(neighbor_agg)
            quad_edges.append(edge)


        quad_neighbors = tf.concat(quad_neighbors, axis=-1)
        x = tf.concat((x, quad_neighbors), axis=-1)

        if self.use_global:
            _global = tf.tile(graph['global'], [tf.shape(x)[0], 1])
            x = tf.concat((x, _global), axis=-1)

        x = self.node_transform(x)
        x = self.activation(x)

        graph['edges'] = quad_edges
        
        if self.use_global:
            edge_global = edge_global * 1./(self.global_scale_edges * 300. * 300. * 8.)
            node_global = tf.reduce_sum(x, axis=0, keepdims=True)
            node_global = node_global * 1./(self.global_scale_nodes * 300. * 300.)
            global_in = tf.concat((graph['global'], node_global, edge_global), axis=-1)
            graph['global'] = self.glob_transform(global_in)
        
        return x, graph
    

class GraphDownsamplingBlock(tf.keras.layers.Layer):
    
    def __init__(self, units, activation=None, use_bias=False, n_layers=1, diffuse_graph=True, 
        residual=False, layer_type=None, use_global=False):
        super(GraphDownsamplingBlock, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.diffuse_graph = diffuse_graph
        self.residual = residual
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.use_global = use_global

    def build(self, input_shape):
        GraphLayer = L.get(self.layer_type, GeoQuadrantGCN)
        self.gcns = [ GraphLayer(self.units, activation=self.activation, use_bias=self.use_bias, 
                                 use_global=self.use_global) for l in range(self.n_layers) ]

    #@tf.function
    def call(self, x, graph):
        n_nodes = tf.shape(x)[0]
        res = x

        # apply convolution
        x, graph = self.gcns[0](x, graph)
        for gcn in self.gcns[1:]:
            if self.diffuse_graph:
                graph['edge_index'] = graph_ops.diffuse(graph['edge_index'], n_nodes)
                x, graph = gcn(x, graph)
            else:
                x, graph = gcn(x, graph)

        if self.residual:
            x = tf.concat((x, res), axis=-1)

        if DEBUGGING:
            self.outputs = {
                'input_graph' : graph['edge_index'],
                'input_node_loc' : graph['node_loc'],
            }

        # apply max pooling
        x, graph = graph_pooling_2x2(x, graph)

        if DEBUGGING:
            self.outputs.update({
                'target_graph' : graph['edge_index'],
                'target_node_loc' : graph['node_loc'],
            })

        return x, graph

class GraphUpsamplingBlock(tf.keras.layers.Layer):
    
    def __init__(self, units, activation=None, use_bias=False, n_layers=2, diffuse_graph=True, residual=False, 
        layer_type=None, last_block=False, use_global=False):
        super(GraphUpsamplingBlock, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.diffuse_graph = diffuse_graph
        self.residual = residual
        self.n_layers = n_layers
        self.last_block = last_block
        self.layer_type = layer_type
        self.use_global = use_global

    def build(self, input_shape):
        self.upsampling = GraphUpsampling2x2(self.units, method='graph', 
            layer_type=self.layer_type, use_bias=self.use_bias, 
            activation=self.activation, use_global=self.use_global)
        self.gcns = []
        GraphLayer = L.get(self.layer_type, GeoQuadrantGCN)

        for l in range(self.n_layers):
            self.gcns.append(GraphLayer(self.units, activation=self.activation, 
                use_bias=self.use_bias, learn_global_scale=not(self.last_block and (l == self.n_layers-1)), 
                use_global=self.use_global))

    #@tf.function
    def call(self, x, graph, x_skip, target_graph):
        if DEBUGGING:
            x = tf.range(100, 100+tf.shape(x)[0], dtype=tf.float32)[:,tf.newaxis]
            
            self.outputs = {
                'target_graph' : target_graph['edge_index'],
                'input_graph' : graph['edge_index'],
                'x_input' : x,
                'target_node_loc' : target_graph['node_loc'],
                'input_node_loc' : graph['node_loc'],
            }
        x, graph = self.upsampling(x, graph, target_graph)
        x_res = x
        # concat skip connection
        x = tf.concat((x, x_skip), axis=-1)
        # apply convolution
        x, graph = self.gcns[0](x, graph)

        for gcn in self.gcns[1:]:
            if self.diffuse_graph:
                edge_index = graph['edge_index']
                graph['edge_index'] = graph_ops.diffuse(edge_index, n_nodes)
                x, graph = gcn(x, graph)
                # undo the diffusion
                graph['edge_index'] = edge_index
            else:
                x, graph = gcn(x, graph)

        if self.residual:
            x = x + x_res

        return x, graph

class GraphUpsampling2x2(tf.keras.layers.Layer):    

    def __init__(self, units=None, method='constant', col_idx=0, layer_type=None, use_global=False, use_bias=False, activation=None):
        super(GraphUpsampling2x2, self).__init__()
        self.col_idx = col_idx
        self.pad = [tf.eye(2, dtype=tf.int32)[-col_idx-1]]
        self.method = method
        self.units = units
        self.use_bias = use_bias
        if self.method == 'graph':
            GraphLayer = L.get(layer_type, GeoQuadrantGCN)
            self.gcn = GraphLayer(self.units, activation=activation, use_bias=use_bias, use_global=use_global)

    def call(self, x, graph, target_graph):
        # get indices of the inserted nodes, i.e. nodes at a position
        # with an odd value
        upsampled_pos = target_graph['node_loc']
        # get number of columns of the node grid
        n_cols = tf.reduce_max(upsampled_pos[:,self.col_idx], axis=-1, keepdims=True) + 1
        padding = tf.pad(n_cols, self.pad, constant_values=1)
        half_pos = upsampled_pos // 2 * padding
        pos_id = tf.reduce_sum(half_pos, axis=-1)
        unique_node_bins = tf.unique(pos_id)
        # new node ids maps the index of input x onto the
        # upsampled version of the graph (max 4 neighbors)
        new_node_ids = unique_node_bins.idx
        n_pooled_nodes = tf.shape(unique_node_bins.y)[0]

        def gather_edges(edges, new_nodes_hot):
            new_receivers_hot = tf.gather(new_nodes_hot, edges[:,1])
            new_nodes_edges = tf.boolean_mask(edges, new_receivers_hot)
            return new_nodes_edges
        
        # nearest neighbor upsampling
        if self.method == 'nearest':
            x = tf.gather(x, new_node_ids)
        elif self.method == 'constant':
            # insert new nodes and keep old nodes
            old_node_ids = tf.math.unsorted_segment_min(
                tf.range(tf.shape(new_node_ids)[0]), 
                new_node_ids, 
                n_pooled_nodes)[:,tf.newaxis]
            x = tf.scatter_nd(tf.cast(old_node_ids, tf.int32), x, 
                tf.concat((tf.shape(upsampled_pos)[:1], x.shape[1:]), axis=0))
        elif self.method == 'graph_naive':
            '''
            Uses the target graph for upsampling
            '''
            # insert new (constant) nodes and keep old nodes
            old_node_ids = tf.math.unsorted_segment_min(
                tf.range(tf.shape(new_node_ids)[0]), 
                new_node_ids, 
                n_pooled_nodes)[:,tf.newaxis]
            old_nodes_hot = tf.scatter_nd(
                tf.cast(old_node_ids, tf.int32), 
                tf.ones(tf.shape(x)[0]), 
                shape=tf.shape(upsampled_pos)[:1])
            x = tf.scatter_nd(tf.cast(old_node_ids, tf.int32), x, 
                shape=tf.concat((tf.shape(upsampled_pos)[:1], x.shape[1:]), axis=0))
            new_nodes_hot = 1 - old_nodes_hot

            # get all edges ending in a new node
            if tf.nest.is_nested(graph['edge_index']):
                new_nodes_edges = tf.nest.map_structure(lambda e: gather_edges(e, new_nodes_hot), graph['edge_index'])
            else:
                new_nodes_edges = gather_edges(target_graph['edge_index'], new_nodes_hot)
            
            graph['edge_index'] = new_nodes_edges
            graph['edges'] = tf.nest.map_structure(lambda e: 
                tf.zeros_like(e, dtype=tf.float32), new_nodes_edges)
            graph['global'] = tf.concat((graph['global'], target_graph['global']), axis=-1)
            x, graph = self.gcn(x, graph)
            #self.outputs['x_out'] = x
            graph['edge_index'] = target_graph['edge_index']
            # ToDo: use the edge features from the gcn
            graph['edges'] = target_graph['edges']
            graph['node_loc'] = target_graph['node_loc']
        elif self.method == 'graph':
            '''
            Uses a new graph for upsampling connecting existing nodes to their upsampled neigbors
            (max 4 neighbors)
            '''
            in_nodes_shape = tf.shape(x)
            out_nodes_shape = tf.shape(upsampled_pos)
            senders = new_node_ids
            receivers = tf.range(in_nodes_shape[0], in_nodes_shape[0] + out_nodes_shape[0])
            new_nodes_edges = tf.stack((senders, receivers), axis=-1)
            # create new nodes
            x_zero = tf.zeros(tf.concat((out_nodes_shape[:1], in_nodes_shape[1:]), axis=0))
            # concat existing nodes
            x = tf.concat((x, x_zero), axis=0)

            # get all edges ending in a new node
            if tf.nest.is_nested(graph['edge_index']):
                # divide edges into four edge-lists to use with QuadGCN
                old_locs = tf.cast(graph['node_loc'], tf.float32)*2+0.5
                pos = tf.concat((old_locs, tf.cast(upsampled_pos, tf.float32)), axis=0)
                rel_pos = tf.gather(pos, new_nodes_edges)
                rel_pos = rel_pos[:,1] - rel_pos[:,0]
                cond = [[-.5, .5], [.5, .5], [.5, -.5], [-.5, -.5]]
                #tf.print('#', new_nodes_edges, rel_pos, cond[0], tf.reduce_all(tf.equal(rel_pos, cond[:1]), axis=-1))
                new_nodes_edges = [ tf.boolean_mask(
                    new_nodes_edges, tf.reduce_all(tf.equal(rel_pos, cond[i]), axis=-1)) \
                                    for i in range(4) ]

            e_index_orig = graph['edge_index']

            if DEBUGGING:
                self.outputs = {
                    'us_x_in' : x,
                    'new_nodes_hot' : tf.concat((tf.ones(in_nodes_shape[0]), 
                                                 tf.zeros(out_nodes_shape[0])), axis=0),
                    'upsampled_pos' : upsampled_pos,
                    'new_nodes_edges' : new_nodes_edges,
                    'scaled_node_pos' : old_locs
                }

            graph['edge_index'] = new_nodes_edges
            # new_nodes_edges: [[3 2], [3 2], [1 2], [1 2]]
            graph['edges'] = tf.nest.map_structure(lambda e: 
                tf.zeros_like(e, dtype=tf.float32), new_nodes_edges)
            graph['global'] = tf.concat((graph['global'], target_graph['global']), axis=-1)
            x, graph = self.gcn(x, graph)
            x = x[in_nodes_shape[0]:]

            if DEBUGGING:
                self.outputs['x_out'] = x

            graph['edge_index'] = target_graph['edge_index']
            graph['edges'] = target_graph['edges']
            graph['node_loc'] = target_graph['node_loc']

        else:
            raise NotImplementedError(f'Upsampling method "{self.method}" is not implemented.')
        
        return x, graph

def remove_duplicate_edges(edge_index, n_nodes, edges, remove_selfloops=True):
    # there might be double edges due to the diffusion on the complete graph
    # remove duplicates:
    dtype = edge_index.dtype
    edge_index = tf.cast(edge_index, tf.int64)
    lin_edges = edge_index * tf.pad(n_nodes[tf.newaxis], [[0, 1]], constant_values=1)
    edge_unique = tf.unique(tf.reduce_sum(lin_edges, axis=-1))
    new_edges = edge_unique.y
    new_edges = tf.stack((new_edges // n_nodes, new_edges % n_nodes), axis=-1)

    new_edge_features = tf.math.unsorted_segment_max(edges, edge_unique.idx, tf.shape(new_edges)[0])

    if remove_selfloops:
        non_selfloop = tf.not_equal(new_edges[:,0], new_edges[:,1])
        new_edges = tf.boolean_mask(new_edges, non_selfloop)
        new_edge_features = tf.boolean_mask(new_edge_features, non_selfloop)

    return tf.cast(new_edges, dtype), new_edge_features

def graph_pooling_2x2(x, graph, col_idx=0):
    edge_index = graph['edge_index']
    edge_features = graph['edges']
    node_loc = graph['node_loc']

    n_nodes = tf.cast(tf.shape(node_loc)[0], tf.int64)
    # pad one for the row idx
    pad = [tf.eye(2, dtype=tf.int32)[-col_idx-1]]
    
    half_cols = tf.reduce_max(node_loc[:,col_idx], axis=-1, keepdims=True) + 1
    padding = tf.pad(half_cols, pad, constant_values=1)
    half_pos = node_loc // 2 * padding
    pos_id = tf.reduce_sum(half_pos, axis=-1)
    unique_node_bins = tf.unique(pos_id)
    new_node_ids = unique_node_bins.idx
    n_pooled_nodes = tf.shape(unique_node_bins.y)[0]
    
    x_pooled = tf.math.unsorted_segment_max(x, new_node_ids, n_pooled_nodes)
    new_pos_ext = node_loc // 2
    new_pos = tf.math.unsorted_segment_min(new_pos_ext, new_node_ids, n_pooled_nodes)
    
    def transform_edges(e_index, edges):
        e_index = tf.gather(new_node_ids, e_index)
        e_index, edges = remove_duplicate_edges(e_index, n_nodes, edges)
        return e_index, edges
    
    if tf.nest.is_nested(edge_index):
        new_edges = tf.nest.map_structure(transform_edges, edge_index, edge_features)
        new_edges = tf.nest.flatten(new_edges)
        new_e_index = new_edges[::2]
        new_e_features = new_edges[1::2]
    else:
        new_e_index, new_e_features = transform_edges(edge_index, edge_features)
    
    graph['edge_index'] = new_e_index
    graph['edges'] = new_e_features
    graph['node_loc'] = new_pos
    return x_pooled, graph

def graph_upsampling_2x2(x, node_loc, upsampled_pos, col_idx=0, method='constant'):    
    pad = [tf.eye(2, dtype=tf.int32)[-col_idx-1]]
    # get indices of the inserted nodes, i.e. nodes at a position
    # with an odd value
    half_cols = tf.reduce_max(upsampled_pos[:,col_idx], axis=-1, keepdims=True) + 1
    padding = tf.pad(half_cols, pad, constant_values=1)
    half_pos = upsampled_pos // 2 * padding
    pos_id = tf.reduce_sum(half_pos, axis=-1)
    unique_node_bins = tf.unique(pos_id)
    new_node_ids = unique_node_bins.idx
    n_pooled_nodes = tf.shape(unique_node_bins.y)[0]
    
    # nearest neighbor upsampling
    if method == 'nearest':
        x = tf.gather(x, new_node_ids)
    elif method == 'constant':
        # insert new nodes and keep old nodes
        old_node_ids = tf.math.unsorted_segment_min(
            tf.range(tf.shape(new_node_ids)[0]), 
            new_node_ids, 
            n_pooled_nodes)
        x = tf.scatter_nd(tf.cast(old_node_ids[:,tf.newaxis], tf.int32), x, 
                          tf.concat((tf.shape(upsampled_pos)[:1], x.shape[1:]), axis=0))
    else:
        raise NotImplementedError(f'Upsampling method "{method}" is not implemented.')
    
    return x