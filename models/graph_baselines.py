import tensorflow as tf
import tensorflow.keras.layers as layers
import layers as L
import layers.graph_ops as graph_ops
from layers.graph_layers import GCN, GAT, MPNN
from models.base_model import GraphBaseModel


class GraphResNet(GraphBaseModel):
	'''
	Implementation of GResNet (2019). A residual architecture using
	the vanilla spatial graph convolution.
	https://arxiv.org/pdf/1909.05729.pdf
	'''

	def __init__(self, units, out_units=8, depth=5, layer_type='gat', output_activation=None, **layer_args):
		super(GraphResNet, self).__init__()

		GraphLayer = L.get(layer_type)
		res_units = units

		if layer_type == 'gat':
			if not layer_args:
				layer_args['heads'] = 3
			res_units = units // layer_args['heads'] * layer_args['heads']
			units = units // layer_args['heads']

		self.blocks = []
		self.res_linear = layers.Dense(res_units, use_bias=False)
		self.res_output = layers.Dense(out_units, use_bias=False)

		for i in range(depth - 1):
			gnn = GraphLayer(units, **layer_args)
			gcn_res = GCN(0, use_weight=False, use_bias=False)
			bn = layers.BatchNormalization()

			self.blocks.append((gnn, gcn_res, bn))

		gnn = GraphLayer(out_units, **layer_args)
		gcn_res = GCN(0, use_weight=False, use_bias=False)
		self.blocks.append((gnn, gcn_res))
		self.layer_type = layer_type
		self.output_activation = layers.Activation(output_activation)

	def call(self, x, edge_index, extra_features=None, training=False):

		res = self.res_linear(x)

		if self.layer_type == 'geo_quadrant_gcn':
			res_edge_index = tf.concat(edge_index, axis=0)
		else:
			res_edge_index = edge_index

		for gnn, gcn_res, bn in self.blocks[:-1]:
			x = gnn(x, edge_index)
			x = x + gcn_res(res, res_edge_index)
			x = bn(x, training=training)
			x = tf.nn.relu(x)

		#x = tf.concat((x_input, x), axis=-1)
		#x = self.out_gnn(x, edge_index)
		gnn, gcn_res = self.blocks[-1]

		if self.layer_type == 'gat':
			x = gnn(x, edge_index, concat_heads=False)
		else:
			x = gnn(x, edge_index)
			
		x = x + gcn_res(self.res_output(res), res_edge_index)

		return self.output_activation(x)


class GATBaseline(GraphBaseModel):
	'''
	Implementation of a pure GAT.
	'''
	def __init__(self, units, out_units=8, depth=3, activation='elu', use_bias=False, heads=3, 
		output_activation=None, **kwargs):
		super(GATBaseline, self).__init__()
		self.gat_layers = [ GAT(units, heads, activation, use_bias=use_bias) for l in range(depth-1) ]
		self.gat_layers.append(GAT(out_units, heads, use_bias=use_bias))
		self.activation = layers.Activation(activation)
		self.output_activation = layers.Activation(output_activation)

	def call(self, x, edge_index, extra_features=None, training=False):

		for gnn in self.gat_layers[:-1]:
			x = gnn(x, edge_index, training=training, concat_heads=True)
			x = self.activation(x)

		x = self.gat_layers[-1](x, edge_index, training=training, concat_heads=False)
		
		return self.output_activation(x)


class GNN(GraphBaseModel):
	'''
	Implementation of a generic graph model with the specified layer operation.
	https://arxiv.org/pdf/1909.05729.pdf
	'''

	def __init__(self, units, out_units=8, depth=5, activation=None, layer_type=None, 
		output_activation=None, **layer_args):
		super(GNN, self).__init__()
		if layer_type is None:
			GraphLayer = MPNN
		else:
			GraphLayer = L.get(layer_type)

		self.gnn_layers = [ GraphLayer(units, **layer_args) for l in range(depth-1) ]
		self.gnn_layers.append(GraphLayer(out_units, **layer_args))
		self.activation = layers.Activation(activation)
		self.output_activation = layers.Activation(output_activation)
		self.layer_type = layer_type

	def call(self, x, edge_index, extra_features=None, training=False):

		x = self.gnn_layers[0](x, edge_index, training=training)

		for gnn in self.gnn_layers[1:]:
			x = self.activation(x)
			x = gnn(x, edge_index, training=training)

		return self.output_activation(x)


class SkipGNN(GraphBaseModel):
	'''
	GNN with skip connections and output 2-layer MLP.
	This is a multilayer GNN in a Wavenet-style setup.
	'''

	def __init__(self, units, out_units=8, depth=5, activation=None, layer_type=None, 
		output_activation=None, **layer_args):
		super(SkipGNN, self).__init__()
		if layer_type is None:
			GraphLayer = MPNN
		else:
			GraphLayer = L.get(layer_type)

		layer_units = units
		output_activation = activation

		if activation == 'gated':
			layer_units = units * 2
			self.activation = self.gated_activation
		else:
			self.activation = layers.Activation(activation)

		self.depth = depth
		self.in_layer = layers.Dense(units)
		self.gnn_layers = [ GraphLayer(layer_units, **layer_args) for l in range(depth) ]
		self.skip_layers = [ layers.Dense(units) for l in range(depth) ]
		self.block_layers = [ layers.Dense(units) for l in range(depth-1) ]
		self.block_layers.append(lambda x: x)
		self.layer_type = layer_type
		self.out_model = tf.keras.Sequential([
			layers.Activation(activation),
			layers.Dense(units),
			layers.Activation(activation),
			layers.Dense(out_units, use_bias=False, activation=output_activation)
		])

	def gated_activation(self, x):
		gate, activation = tf.split(x, 2, axis=-1)
		gate = tf.nn.sigmoid(gate)
		activation = tf.nn.relu(activation)
		return gate * activation

	'''def diffuse_graph(self, edge_index, n_nodes):

		diffuse = lambda x: tf.gather(
			graph_ops.get_khop_index(x, 
				k=2**self.depth, 
				n_nodes=n_nodes, 
				output_intermediate_graphs=True),
			2**tf.range(k) - 1)

		if tf.nest.is_nested(edge_index):
			return zip(*tf.nest.map_structure(diffuse, edge_index))
		else:
			return diffuse(edge_index)

	def call(self, x, edge_index, extra_features=None, training=False):

		skip_out = 0
		edge_index = self.diffuse_graph(edge_index, x.shape[0])

		for gnn, skip_layer, e_index in zip(self.gnn_layers, self.skip_layers, edge_index):
			x = gnn(x, e_index, training=training)
			skip_out += skip_layer(x)
			x = self.activation(x)

		skip_out = self.out_model(skip_out)

		return skip_out'''




	def diffuse_graph(self, edge_index, n_nodes):

		diffuse = lambda x: tf.gather(
			graph_ops.get_khop_index(x, 
				k=2**self.depth, 
				n_nodes=n_nodes, 
				output_intermediate_graphs=True),
			2**tf.range(k) - 1)

		if tf.nest.is_nested(edge_index):
			return zip(*tf.nest.map_structure(diffuse, edge_index))
		else:
			return diffuse(edge_index)

	def diffuse_and_traverse(self, edge_index, logits, n_nodes, n_neighbors=1):
		edge_index1 = graph_ops.sample_neighbors(edge_index, 1, logits, n_nodes)
		edge_index2 = graph_ops.sample_neighbors(edge_index, 1, logits, n_nodes)
		unique_edges = tf.minimum(1, tf.reduce_sum(tf.abs(edge_index1 - edge_index2), axis=-1))
		edge_index2 = tf.boolean_mask(edge_index2, unique_edges)
		edge_index = tf.concat((edge_index1, edge_index2), axis=0)
		edge_index = graph_ops.diffuse(edge_index, n_nodes)
		edge_index = tf.concat((edge_index, tf.tile(tf.range(n_nodes, dtype=edge_index.dtype)[:,tf.newaxis], [1,2])), axis=0)
		return edge_index

	@tf.function
	def call(self, x, edge_index, extra_features=None, training=False):

		skip_out = 0
		n_nodes = x.shape[0]
		e_index = edge_index
		node_logits = tf.math.log(extra_features['street_size'] + 1e-8)
		e_index_sl = tf.nest.map_structure(lambda x: 
				tf.concat((x, tf.tile(tf.range(n_nodes, dtype=x.dtype)[:,tf.newaxis], [1,2])), axis=0), 
				e_index)

		x = self.in_layer(x)

		for d, (gnn, skip_layer, dense) in enumerate(zip(self.gnn_layers, self.skip_layers, self.block_layers)):
			#for i in range(2**max(0, d-1), 2**d):
			#	#e_index = tf.nest.map_structure(
			##	#	lambda j, u: 
			#	#		graph_ops.adj2index(
			#	#			graph_ops.index2adj(j, n_nodes, tf.int8) @ graph_ops.index2adj(u, n_nodes, tf.int8)), 
			#	#	e_index, edge_index)
			res = x
			x = gnn(x, e_index_sl, training=training)

			e_index_sl = tf.nest.map_structure(lambda x: 
				self.diffuse_and_traverse(x, node_logits, n_nodes, 1), 
				e_index)
			e_index = e_index_sl[:-n_nodes]

			x = self.activation(x)
			skip_out += skip_layer(x)
			x = dense(x)
			x += res

		skip_out = self.out_model(skip_out)

		return skip_out