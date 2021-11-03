import tensorflow as tf
import data.data_utils as data_utils
import data
import evaluation
import numpy as np

# -------------------Code to setup the training pipeline -----------------------

def build_data_pipeline(t4c, config):
    '''
    Builds the pipeline for the tensorflow datasets-.
    '''

    PI = tf.constant(np.pi)

    def sample_quadrant_subgraphs(x):
        '''
        Splits the graph into subgraphs by quadrant direction 
        (s. data_utils.sample_quadrant_subgraphs).
        '''
        x['graph']['edge_index'] = data_utils.sample_quadrant_subgraphs(
            x['graph']['edge_index'], 
            x['graph']['edge_direction'])
        return x

    def add_node_pos(x):
        '''
        Adds the 2D node location (position in the pixel grid) to the
        graph dictionary.
        '''
        c_idx = t4c.city_lookup.lookup(x['city'])
        c_idx.set_shape(1)
        node_loc = tf.gather(t4c.city_node_loc, c_idx)[0].to_tensor()
        x['graph']['node_loc'] = node_loc
        return x

    def add_street_map(x):
        '''
        Adds the street map (1st channel of the static street data to the
        image dictionary.
        '''
        c_idx = t4c.city_lookup.lookup(x['city'])
        c_idx.set_shape(1)
        street_map = tf.gather(t4c.city_street_map, c_idx)[0]
        x['image'] = x.get('image', {})
        x['image']['street_map'] = street_map
        return x

    def add_temp_encoding(x):
        '''
        Generates an encoding of daytime and concatenates it to every node.
        Alternatively this could also be treated as a global variable, 
        e.g. as a 'master' node.
        '''
        # get 5-min indices over one hour
        ts = tf.range(12, dtype=tf.float32)[tf.newaxis]
        # translate by starting time
        ts += tf.cast(x['time_idx'], tf.float32)[:,tf.newaxis]
        # normalize and convert to radians
        ts = ts / (12.*24.) * 2. * PI
        t_enc = tf.stack((tf.math.sin(ts + 12.), tf.math.cos(ts)), axis=-1)
        x['time_enc'] = t_enc
        return x
    
    if config['data_type'] == 'image':
        t4c = t4c.as_images(config['seed_len'], config['target_len'])
        t4c = t4c.batch(config['batch'])
    else:
        if config['batch'] == 1:
            t4c = t4c.as_graphs(config['seed_len'], config['target_len'])
        else:
            t4c = t4c.as_batched_graphs(config['batch'], config['seed_len'], config['target_len'])

    if config['add_temp_encoding']:
        t4c.add_transform(add_temp_encoding)

    if config['layer_type'] == 'geo_quadrant_gcn':
        t4c.add_transform(sample_quadrant_subgraphs)

    if 'GraphUNet' in config['model']:
        t4c.add_transform(add_node_pos)
        t4c.add_transform(add_street_map)

    t4c.add_transform(t4c.concat_time)
    return t4c

# --------------------------- Utility ---------------------------

def flatten_ragged(ragged, limit=None):
    '''
    Flattens the nested tensors in ragged into a single vector.
    Used to flatten the gradient for histogram plots.

    limit : int
        Number of scalars to include from each nested tensor.
    '''
    flattened = tf.concat(
        tf.nest.flatten(
            tf.nest.map_structure(
                lambda v: tf.reshape(v, [-1])[:limit], 
                ragged)), 
        axis=0)
    return flattened

def create_evaluation_plots(target, prediction, model_input, mask, input_timesteps=12, target_timesteps=12):
    '''
    Generates evaluation plots for graph data and image data.
    '''
    prediction_ndims = prediction.shape.ndims

    is_graph = prediction_ndims <= 3
    is_temporal = prediction_ndims == 3 or prediction_ndims == 5

    if is_graph:
        model_input = data_utils.graph_to_image(model_input, mask)
        prediction = data_utils.graph_to_image(prediction, mask)
        target = data_utils.graph_to_image(target, mask)

    if not is_temporal:
        model_input = data_utils.unstack_time(model_input, input_timesteps, axis=-4)
        prediction = data_utils.unstack_time(prediction, input_timesteps, axis=-4)
        target = data_utils.unstack_time(target, target_timesteps, axis=-4)

    # create training plots
    train_plots = evaluation.create_eval_plots(
        target.numpy(), 
        prediction.numpy(), 
        x=model_input[...,:8].numpy(), 
        mask=mask.numpy())

    return train_plots

# ------------------------ Learning rate schedules ------------------------

def warmupExpDecay(step, warum_steps=2000., warmup_max=0.002, decay=0.98, 
    decay_min=0.0002, decay_every=100.):
    step = tf.cast(step, tf.float32)
    a = (warmup_max-decay_min) / (decay**(warum_steps//decay_every))
    return tf.minimum(warmup_max/warum_steps * step, decay_min+a*decay**(step//decay_every))
