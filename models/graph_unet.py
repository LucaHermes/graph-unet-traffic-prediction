import tensorflow as tf
import tensorflow.keras.layers as layers
import layers as L
import layers.graph_ops as graph_ops
from layers.graph_layers import GCN, GAT, MPNN
from layers.hybrid_gnn import GraphDownsamplingBlock, GraphUpsamplingBlock, GeoQuadrantGCN
from models.base_model import GraphBaseModel

class GraphUNet(GraphBaseModel):
	'''
	This model might benefit from:
	 * explicit temporal processing (1d-convs a la WaveNet?)
	 * graph diffusion
	 * global traffic information (global nodes?)
	     * before we do this, we should improve toward a better temporal modelling
	     * when we do this, we should make sure that the nodes know if they 
	       are in the center or at the edge of the traffic map
	 * autoregressive forecasting?
	'''

	def __init__(self, units, out_units=8, depth=5, activation=None, layer_type=None, use_bias=False, 
		use_global=False, output_activation=None, **layer_args):
		super(GraphUNet, self).__init__()

		self.depth = depth
		self.downsampling_layers = []
		self.upsampling_layers = []
		self.use_global = use_global
		self.activation = activation
		self.layer_type = layer_type
		self.use_bias = use_bias
		layer_units = units
		GraphLayer = L.get(layer_type)
		self.endpoints = []
		print('##### Output activation:', output_activation)
		print('##### Using bias:       ', use_bias)
		self.in_layer = layers.Dense(max(layer_units, 128))
		
		for i in range(self.depth):
			print('Create Downsampling block: ', max(layer_units, 128), 'units')
			ds = GraphDownsamplingBlock(max(layer_units, 128), activation=activation, use_bias=use_bias, diffuse_graph=False, 
				layer_type=layer_type, use_global=use_global)
			layer_units = units * 2**(i+1)
			self.downsampling_layers.append(ds)

		self.bottleneck_gcn_1 = GraphLayer(128, activation=self.activation, use_bias=self.use_bias, 
			learn_global_scale=self.use_global, use_global=self.use_global)
		self.bottleneck_gcn_2 = GraphLayer(128, activation=self.activation, use_bias=self.use_bias, 
			learn_global_scale=self.use_global, use_global=self.use_global)

		for i in range(self.depth):
			layer_units = units * 2**(self.depth-i-1)
			print('Create Upsampling block:   ', max(layer_units, 128), 'units')
			us = GraphUpsamplingBlock(max(layer_units, 128), activation=activation, use_bias=use_bias, diffuse_graph=False, 
				layer_type=layer_type, use_global=use_global, last_block=i == (self.depth-1))
			self.upsampling_layers.append(us)

		self.out_model = layers.Dense(out_units, use_bias=False, activation=output_activation)
		
	def call(self, x, graph, training=False):
		n_nodes = tf.shape(x)[0]

		skip_graphs = []
		skips = []

		x = self.in_layer(x)

		for d, layer in enumerate(self.downsampling_layers):
			skip_graphs.append(graph.copy())
			skips.append(x)
			x, graph = layer(x, graph)

		x, graph = self.bottleneck_gcn_1(x, graph)
		x, graph = self.bottleneck_gcn_2(x, graph)

		di = 0

		for d, layer in enumerate(self.upsampling_layers):
			x, graph = layer(x, graph, skips[-(di+1)], skip_graphs[-(di+1)]) # x, e_index, node_loc, upsampling_node_loc, skip)
			di += 1

		x = self.out_model(x)

		return x, graph
