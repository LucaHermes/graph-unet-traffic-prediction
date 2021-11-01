from datetime import datetime
import tensorflow as tf
import os
import wandb
import train_utils
import evaluation
import data.data_utils as data_utils
import data
from collections import namedtuple

TSPEC_D = [{
	'graph': {
		'seed_nodes': tf.TensorSpec(shape=(None, 96), dtype=tf.float32, name=None),
		'target_nodes': tf.TensorSpec(shape=(None, 96), dtype=tf.float32, name=None),
		'edge_index': [tf.TensorSpec(shape=(None, 2), dtype=tf.int32, name=None)]*4,
		'edges': tf.TensorSpec(shape=(None,), dtype=tf.float32, name=None),
		'edge_direction': tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None),
		'node_loc': tf.TensorSpec(shape=(None, 2), dtype=tf.int64, name=None)
	},
   'city': tf.TensorSpec(shape=(1,), dtype=tf.string, name=None),
   'date': tf.TensorSpec(shape=(1,), dtype=tf.string, name=None),
   'time_idx': tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None),
   'time_enc': tf.TensorSpec(shape=(1,12,2), dtype=tf.float32, name=None),
   'weekday': tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None),
   'image': {
	   'street_map': tf.TensorSpec(shape=(495, 436), dtype=tf.int32, name=None)
   }
}]

class BaseModel(tf.keras.Model):
	'''
	Tensorflow base model. This base model contains:
	 * logger varables (global_step, train_epoch, train_logs, val_logs)
	 * checkpointing after every epoch
	'''
	def __init__(self, *args, **kwargs):
		super(BaseModel, self).__init__(*args, **kwargs)
		self.train_epoch = tf.Variable(0, trainable=False)
		self.global_step = tf.Variable(0, trainable=False)
		
		self.architecture = str(self.__class__.__name__)
		date = datetime.now().strftime('%d-%m-%Y__%H-%M-%S')
		self._id = tf.Variable('%s_%s' % (self.architecture, date), 
			trainable=False)

		self.train_logs = {}
		self.val_logs = {}

	@property
	def id(self):
		'''
		Id is a combination of class name and
		date of creation. Its a tf Variable, so it gets saved and
		loaded via tf save/load methods.
		'''
		return self._id.numpy().decode()

	def save(self, directory, optimizer=None):
		ckpt = tf.train.Checkpoint(
			step=self.global_step, 
			optimizer=optimizer, 
			net=self)
		ckpt_manager = tf.train.CheckpointManager(
			ckpt, 
			directory, 
			max_to_keep=3)
		return ckpt_manager.save()

	def load(self, directory, optimizer=None):
		ckpt = tf.train.Checkpoint(
			step=self.global_step, 
			optimizer=optimizer, 
			net=self)
		ckpt_manager = tf.train.CheckpointManager(
			ckpt, 
			directory, 
			max_to_keep=3)

		if ckpt_manager.latest_checkpoint:
			print('Restoring model from checkpoint', ckpt_manager.latest_checkpoint)
			return ckpt_manager.restore_or_initialize()

	@tf.function
	def quantitative_evaluation_step(self, x):
		pred = self._call(x, training=False)
		return self._compute_metrics(x, pred)


	def quantitative_evaluation(self, dataset):
		mean_tensor = tf.keras.metrics.MeanTensor()

		for traffic_data in dataset:
			metrics_dict = self.quantitative_evaluation_step(traffic_data)
			mean_tensor.update_state(list(metrics_dict.values()))

		metrics = mean_tensor.result()
		val_keys = [ 'val_%s' % k for k in metrics_dict.keys() ]
		val_metrics = dict(zip(val_keys, metrics))
		return val_metrics

	def qualitative_evaluation(self, dataset):
		traffic_data = next(iter(dataset))
		pred = self._call(traffic_data, training=False)

		traffic_input = self.get_seeds(traffic_data)
		target = self.get_targets(traffic_data)
		mask = dataset.get_img_mask(traffic_data['city'])

		val_plots = train_utils.create_evaluation_plots(target, pred, traffic_input, mask)
		return { 'val_%s' % k : wandb.Image(v) for k, v in val_plots.items() }

	def evaluate(self, dataset, group_name=None):
		self.val_logs = {}

		# do quantitative evaluation
		quantitative_results = self.quantitative_evaluation(dataset)

		# do qualitative evaluation
		qualitative_results = self.qualitative_evaluation(dataset)

		self.val_logs.update(quantitative_results)
		self.val_logs.update(qualitative_results)
		if group_name is not None:
			wandb.log({ group_name + '/' + k : v for k, v in self.val_logs.items() })
		else:
			wandb.log(self.val_logs)
		return quantitative_results

	def _compute_metrics(self, traffic_data, pred):
		return self.compute_metrics(traffic_data, pred)

	@tf.function
	def get_gradient_stats(self, gradient):
		flat_gradient = train_utils.flatten_ragged(gradient, limit=100)
		_, global_grad_norm = tf.clip_by_global_norm(gradient, 1e10)
		return {
			'grad' : tf.cast(flat_gradient, tf.float16),
			'grad_norm' : global_grad_norm
		}

	@tf.function
	def train_step(self, traffic_data):
		info = {}

		with tf.GradientTape() as tape:
			pred = self._call(traffic_data, training=True)
			loss = tf.reduce_sum(self.losses)
			
		grad = tape.gradient(loss, self.trainable_variables)
		
		info = { 'pred' : pred, 'loss' : loss, 'gradient' : grad }

		return info

	def train(self, dataset, optimizer, epochs=1, loss_fn=None, log_every=400, save_every=6000,
		ckpts_dir=None, validation_dataset=None, validation_set_sm=None, validation_set_sm_spatial=None, 
		acc_gradients_steps=0):
		'''
		Train the model for multiple epochs.

		dataset : T4CDatasetTF
			Initialized and batched dataset, ready for iteration.
		optimizer : tf.keras.optimizers.Optimizer
			Optimizer used for training.
		epochs : int
			Number of training epochs.
		loss_fn : function
			Function that takes two arguments: ground_truth, prediction
			and returns a loss, defaults to MSE.
		log_every : int
			Specifies the interval in which the evaluation plots 
			(evaluation.create_eval_plots) and extended training statistics 
			are logged. Basic training stats (evaluation.compute_metrics) 
			will be l  ogged after every step.
		'''
		print('Starting trainang %s ...' % self.architecture)

		if loss_fn is None:
			loss_fn = tf.keras.losses.MSE
		if ckpts_dir is None:
			ckpts_dir = 'ckpts'

		if acc_gradients_steps:
			grad_scale = tf.constant(1. / float(acc_gradients_steps))
			_ = self._call(next(iter(dataset)), training=True)
			grads = [ tf.zeros(v.shape) for v in self.trainable_variables ]
			accumulate_grads = lambda grads, delta_grads, reset: tf.nest.map_structure(
					lambda g, dg: 
					(1. - tf.cast(reset, tf.float32)) * g + grad_scale * dg, 
					grads, delta_grads)

		tf.profiler.experimental.server.start(6009)

		for e in range(epochs):
			dataset_it = iter(dataset)
			s = 0
			
			while True:
				self.train_logs = {}
				apply_gradients = (s+1) % max(1, acc_gradients_steps) == 0

				with tf.profiler.experimental.Trace('train', step_num=s):
					try:
						traffic_data = next(dataset_it)
					except StopIteration:
						break
					info = self.train_step(traffic_data)

				if apply_gradients:
					optimizer.apply_gradients(zip(info['gradient'], self.trainable_variables))
					self.global_step.assign_add(1)
				if acc_gradients_steps:
					grads = accumulate_grads(grads, info['gradient'], apply_gradients)
					info['gradient'] = grads

				pred = info['pred']
				self.train_logs.update({
					'loss' : info['loss'],
					'learning_rate' : optimizer.learning_rate,
					'global_step' : self.global_step,
					'epoch' : self.train_epoch,
				})

				if s % 10 == 0:
					self.train_logs.update(self._compute_metrics(traffic_data, pred))

				if s % log_every == 0:
					# create evaluation plots for one training sample
					city = traffic_data['city']
					traffic_input = self.get_seeds(traffic_data)
					target = self.get_targets(traffic_data)
					mask = dataset.get_img_mask(city)
					train_plots = train_utils.create_evaluation_plots(target, pred, traffic_input, mask)
					self.train_logs.update(self.get_gradient_stats(info['gradient']))
					self.train_logs.update({ k : wandb.Image(v) for k, v in train_plots.items() })

				if s % save_every == 0:
					# save the model after every save_every-step
					val_metrics = self.evaluate(validation_set_sm, group_name='train_val')
					val_metrics = self.evaluate(validation_set_sm_spatial, group_name='train_val_spatial')
				if (s+1) % save_every == 0:
					self.save(os.path.join(ckpts_dir, self.architecture, self.id), optimizer=optimizer)
					print('Checkpoint created', datetime.now().isoformat())

				# plot train logs
				wandb.log(self.train_logs)

				s += 1

			if validation_dataset is not None:
				val_metrics = self.evaluate(validation_dataset)

			train_epoch = self.train_epoch.assign_add(1)
			print(f'Epoch {train_epoch-1} finished.')

			# save the model after every epoch
			self.save(os.path.join(ckpts_dir, self.architecture, self.id), optimizer=optimizer)
			
class GraphBaseModel(BaseModel):

	def __init__(self):
		super(GraphBaseModel, self).__init__()
		self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='SAME', activation='relu', use_bias=False)
		self.cnn2 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='SAME', activation='relu', use_bias=False)

	def build(self, input_shape):
		self.quantitative_evaluation_step = tf.function(
			self.quantitative_evaluation_step,
			input_signature=TSPEC_D)
		self.train_step = tf.function(
			self.train_step,
			input_signature=TSPEC_D)

	def forward_street_encoding(self, x):
		# create street encoding by transforming the static data cia 2-layer CNN
		street_encoding = x['image']['street_map'][tf.newaxis, ..., tf.newaxis]
		street_encoding = tf.cast(street_encoding, tf.float32) / 255
		street_encoding = self.cnn2(self.cnn1(street_encoding))[0]
		street_encoding = tf.gather_nd(street_encoding, x['graph']['node_loc'])
		return street_encoding

	def compute_edge_features(self, street_encoding, edge_index):
		edge_encoding = tf.gather(street_encoding, edge_index)
		edge_encoding = tf.concat((edge_encoding[:,0], edge_encoding[:,1]), axis=-1)
		return edge_encoding

	def _call(self, x, training=False):
		traffic_input = self.get_seeds(x)

		edge_index = x['graph']['edge_index']
		
		# create street encoding by transforming the static data cia 2-layer CNN
		street_encoding = self.forward_street_encoding(x)
		if tf.nest.is_nested(edge_index):
			edge_encoding = tf.nest.map_structure(lambda es: 
				self.compute_edge_features(street_encoding, es), edge_index)
		else:
			edge_encoding = self.compute_edge_features(street_encoding, edge_index)
		
		traffic_input = tf.concat((traffic_input, street_encoding), axis=-1)
		
		graph = {
			'edge_index'   : list(edge_index),
			'edges'		: edge_encoding,
			'node_loc'	 : x['graph']['node_loc'],
		}
		if self.use_global:
			global_features = tf.reduce_sum(traffic_input, axis=0, keepdims=True)/(300. * 300.)
			t_enc = x['time_enc'][:,-1]
			weekday_enc = tf.one_hot(x['weekday'], 7)
			global_features = tf.concat((global_features, t_enc, weekday_enc), axis=-1)
			graph['global'] = global_features
		
		pred, graph = self(traffic_input, graph, training=training)

		self.endpoints = []

		if training:
			target = self.get_targets(x)
			mse_loss = tf.reduce_mean(tf.keras.losses.MSE(target, pred))
			self._callable_losses.clear()
			self.add_loss(lambda: mse_loss)
		
		# produce only positive outputs (after the loss calculation)
		return tf.maximum(pred, 0.)

	def get_seeds(self, x):
		return x['graph']['seed_nodes']

	def get_targets(self, x):
		return x['graph']['target_nodes']

	def compute_metrics(self, traffic_data, pred):
		seed = self.get_seeds(traffic_data)
		target = self.get_targets(traffic_data)
		metrics = evaluation.compute_metrics(target, pred, mask=None)
		seed_metrics = evaluation.compute_metrics(target, seed, mask=None, score_only=True)

		img_shape = traffic_data['image']['street_map'].shape
		h, w = img_shape[0], img_shape[1]
		n_nodes = tf.cast(tf.shape(pred)[0], tf.float32)
		n_zeros = tf.cast(h * w, tf.float32) - n_nodes
		score = metrics['mse_scaled']
		score = score * n_nodes/(n_nodes + n_zeros)
		metrics['score'] = score
		
		score = seed_metrics['mse_scaled']
		score = score * n_nodes/(n_nodes + n_zeros)
		metrics['rel_score'] = score / metrics['score']

		return metrics

class ImageBaseModel(BaseModel):

	def __init__(self, **kwargs):
		super(ImageBaseModel, self).__init__(**kwargs)

	def _call(self, x, training=False):
		self._callable_losses.clear()
		traffic_input = self.get_seeds(x)
		target = self.get_targets(x)
		static_data = x['image']['static_data']
		pred = self(traffic_input, static_data, training=training)

		value_loss = tf.reduce_mean(tf.keras.losses.MSE(target, pred))

		self.add_loss(lambda: value_loss)
		
		# produce only positive outputs (after the loss calculation)
		return tf.maximum(pred, 0.)

	def get_seeds(self, x):
		return x['image']['seed_image']

	def get_targets(self, x):
		return x['image']['target_image']

	def compute_metrics(self, traffic_data, pred):
		# pred: [batch, height, width, features]
		city = traffic_data['city']
		# mask: [batch, height, width]
		mask = traffic_data['image']['static_data'][...,0] != 0
		# target: [batch, height, width, features]
		target = self.get_targets(traffic_data)
		seed = self.get_seeds(traffic_data)
		# pred: [batch, height, width, features]
		seed = tf.boolean_mask(seed, mask)
		pred = tf.boolean_mask(pred, mask)
		target = tf.boolean_mask(target, mask)
		metrics = evaluation.compute_metrics(target, pred)
		seed_metrics = evaluation.compute_metrics(target, seed, score_only=True)

		img_shape = tf.shape(mask)
		b, h, w = img_shape[0], img_shape[1], img_shape[2]
		n_nodes = tf.cast(tf.shape(pred)[0], tf.float32)
		n_zeros = tf.cast(b * h * w, tf.float32) - n_nodes
		score = metrics['mse_scaled']
		score = score * n_nodes/(n_nodes + n_zeros)
		metrics['score'] = score

		score = seed_metrics['mse_scaled']
		score = score * n_nodes/(n_nodes + n_zeros)
		metrics['rel_score'] = score / metrics['score']

		return metrics