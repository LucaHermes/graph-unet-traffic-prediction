import tensorflow as tf
import tensorflow.keras.layers as layers

import layers.graph_ops as graph_ops

class GCN(layers.Layer):
	'''
	Implements vanilla graph convolution as presented in this 
	paper: https://arxiv.org/abs/1609.02907
	Effectively, this implements: activation(A' * X  * W),
	where A' is the normalized adjacency matrix, X are the node
	features and W is the weight matrix.
	'''
	def __init__(self, units, activation=None, use_weight=True, use_bias=False, **kwargs):
		super(GCN, self).__init__()
		self.bias = None
		if use_weight:
			self.linear = layers.Dense(units, use_bias=False)
		self.activation = layers.Activation(activation)
		self.use_weight = use_weight
		self.use_bias = use_bias
		self.units = units

	def build(self, input_shape):
		if self.use_bias:
			self.bias = tf.Variable(tf.zeros(self.units), dtype=tf.float32, trainable=True, name='GCN_bias')

	def call(self, x, edge_index):
		x_shape = tf.shape(x)
		n_nodes = x_shape[0]
		n_edges = tf.shape(edge_index)[0]

		node_ext_shape = (n_nodes,) + (1,)*(x.shape.ndims-1)
		edge_ext_shape = (n_edges,) + (1,)*(x.shape.ndims-1)
		# the GCN operation only uses the sender information
		x, _ = graph_ops.get_sender_receiver(x, edge_index)

		# 1. apply sender normalization
		degrees = graph_ops.get_out_degrees(edge_index, expand_edges=True, n_nodes=n_nodes)
		degrees = tf.cast(degrees, tf.float32)
		# ensure there are no zeros to apply the sqrt
		degrees = tf.maximum(1., degrees)
		norm = degrees**-0.5
		#norm = tf.expand_dims(norm, axis=-1)
		norm = tf.reshape(norm, edge_ext_shape)

		# 2. aggregate neighboring features
		x = x * norm
		x = tf.math.unsorted_segment_sum(x, edge_index[:,1], num_segments=n_nodes)

		if self.use_weight:
			x = self.linear(x)

		# 1. apply receiver normalization
		degrees = graph_ops.get_in_degrees(edge_index, expand_edges=False, n_nodes=n_nodes)
		degrees = tf.cast(degrees, tf.float32)
		# ensure there are no zeros to apply the sqrt
		degrees = tf.maximum(1., degrees)
		norm = degrees**-0.5
		#norm = tf.expand_dims(norm, axis=-1)
		norm = tf.reshape(norm, node_ext_shape)
		x = x * norm

		if self.bias is not None:
			x = x + self.bias

		return self.activation(x)

class GAT(layers.Layer):
	'''
	Implements graph attention layer as presented in this 
	paper: Veličković et. al. https://arxiv.org/abs/1710.10903
	'''
	def __init__(self, units, heads=3, activation=None, use_bias=False, **kwargs):
		super(GAT, self).__init__()
		self.linear = layers.Dense(heads * units, use_bias=use_bias)
		w_init = tf.keras.initializers.GlorotNormal()
		self.attention_w1 = tf.Variable(w_init(shape=[heads, 1, units]), trainable=True) #layers.Dense(1, activation=tf.nn.leaky_relu, use_bias=False)
		self.attention_w2 = tf.Variable(w_init(shape=[heads, 1, units]), trainable=True)
		self.activation = layers.Activation(activation)
		self.use_bias = use_bias
		self.heads = heads

	def call(self, x, edge_index, concat_heads=True):
		n_nodes = tf.shape(x)[0]
		n_edges = tf.shape(edge_index)[0]

		# (nodes, features * heads)
		x = self.linear(x)

		sender, receiver = graph_ops.get_sender_receiver(x, edge_index)

		# (heads, edges, features)
		sender = tf.stack(tf.split(sender, self.heads, axis=-1), axis=0)
		# (heads, edges, features)
		receiver = tf.stack(tf.split(receiver, self.heads, axis=-1), axis=0)
		# (heads * edges, 1)
		#tf.concat((sender, receiver), axis=-1)
		#attention = self.attention(attention)
		attention = tf.reduce_sum(self.attention_w1 * sender, axis=-1, keepdims=True) + \
					tf.reduce_sum(self.attention_w2 * receiver, axis=-1, keepdims=True)
		# (heads * edges, 1)		
		attention = tf.reshape(attention, [-1, 1])
		attention = tf.nn.leaky_relu(attention)

		segments = tf.tile(edge_index[:,1], [self.heads])
		segments = segments + tf.repeat(tf.range(self.heads) * n_nodes, n_edges, axis=0)
		# compute softmax over the neighborhoods
		attention = graph_ops.segment_softmax(attention, segments, num_segments=n_nodes*self.heads)

		# compute message
		sender = tf.concat(tf.unstack(sender, axis=0), axis=0)
		sender = sender * attention
		# aggregate messages
		x = tf.math.unsorted_segment_sum(sender, segments, num_segments=n_nodes*self.heads)

		# split heads and concat into feature dimension
		if concat_heads:
			x = tf.concat(tf.split(x, self.heads, axis=0), axis=-1)
		else:
			x = tf.reduce_mean(tf.split(x, self.heads, axis=0), axis=0)

		return self.activation(x)

class MPNN(layers.Layer):
	'''
	Implements a message passing neural network layer inspired by the approach presented in this 
	paper: Gilmer et. al. http://arxiv.org/abs/1704.01212
	'''
	def __init__(self, units, activation=None, use_bias=False, **kwargs):
		super(MPNN, self).__init__()
		self.linear1 = layers.Dense(units, use_bias=use_bias)
		self.linear2 = layers.Dense(units, use_bias=use_bias)
		self.scale = tf.constant(1./9.)
		
		self.activation = layers.Activation(activation)
		self.use_bias = use_bias
		self.units = units

	def build(self, input_shape):
		if input_shape[-1] != self.units:
			self.input_projection = layers.Dense(self.units, use_bias=False)
		else:
			self.input_projection = lambda x: x

	def call(self, x, edge_index):
		n_nodes = tf.shape(x)[0]
		n_edges = tf.shape(edge_index)[0]

		sender, receiver = graph_ops.get_sender_receiver(x, edge_index)

		messages = tf.concat((sender, receiver), axis=-1)
		messages = self.linear1(messages) * self.scale

		x_input = self.input_projection(x)
		x = tf.math.unsorted_segment_sum(messages, edge_index[:,1], num_segments=n_nodes)
		x = tf.concat((x_input, x), axis=-1)

		return self.linear2(self.activation(x))

class GIN(layers.Layer):

	def __init__(self, units, activation=None, use_bias=None, max_neighbors=9, **kwargs):
		super(GIN, self).__init__()
		self.linear1 = layers.Dense(units, use_bias=use_bias)
		#self.scale = tf.constant(1./9.)
		self.activation = layers.Activation(activation)
		self.use_bias = use_bias
		self.units = units
		self.max_neighbors = max_neighbors
		self.bias = None

	def build(self, input_shape):
		#if input_shape[-1] != self.units:
		#	self.input_projection = layers.Dense(self.units, use_bias=False)
		#else:
		#	self.input_projection = lambda x: x

		weight_std = tf.sqrt(2 / ((self.max_neighbors+1) * input_shape[-1] + self.units))
		weight_init = tf.random.truncated_normal([input_shape[-1], self.units], stddev=weight_std)
		self.eps = tf.Variable(tf.zeros([1]), trainable=True, name='GIN_eps')
		self.scale = tf.constant(1./(self.max_neighbors+1), dtype=tf.float32)
		#self.weight_mat = tf.Variable(weight_init, trainable=True, name='GIN_weights')
		#if self.use_bias is not None:
		#	self.bias = tf.Variable(tf.zeros(self.units), dtype=tf.float32, trainable=True, name='GIN_bias')

	def call(self, x, edge_index):
		n_nodes = tf.shape(x)[0]

		sender, receiver = graph_ops.get_sender_receiver(x, edge_index)

		#messages = tf.concat((sender, receiver), axis=-1)
		#x = self.input_projection(x)
		x = (1. + self.eps) * x
		x = x + tf.math.unsorted_segment_sum(sender, edge_index[:,1], num_segments=n_nodes)
		x *= self.scale
		#x = x @ self.weight_mat 

		#if self.use_bias:
		#	x = tf.nn.bias_add(x, self.bias)

		return self.activation(self.linear1(x))


class GraphSerializer:

    def __init__(self, out_nodes, out_edges, nodes_shape):
        self.out_nodes = out_nodes
        self.out_edges = out_edges
        self.nodes_shape = nodes_shape

    def reverse(self, nodes, edge_index, nodes_n_add, edges_n_add):
        return nodes[:-nodes_n_add], edge_index[:-edges_n_add]

    def __call__(self, nodes, edge_index):
        # nodes: [nodes, ...]
        # edge_index: [edges, 2]
        nodes_shape = tf.shape(nodes)
        edge_shape = tf.shape(edge_index)

        nodes_n_add = self.out_nodes - nodes_shape[:1]
        edges_n_add = self.out_edges - edge_shape[0]
        #assert nodes_n_add >= 0
        #assert edges_n_add >= 0

        virtual_shape = tf.concat((nodes_n_add, nodes_shape[1:]), axis=0)

        # add virtual nodes
        virtual_nodes = tf.zeros(virtual_shape)
        nodes = tf.concat((nodes, virtual_nodes), axis=0)

        # add virtual edges
        virtual_edges = tf.ones([edges_n_add, 2], tf.int32) * self.out_nodes-1
        edge_index = tf.concat((edge_index, virtual_edges), axis=0)

        nodes.set_shape([self.out_nodes, 12*8])
        edge_index.set_shape([self.out_edges, 2])

        return nodes, edge_index, nodes_n_add, edges_n_add