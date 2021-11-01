import tensorflow as tf
import numpy as np
import h5py
import os

from datetime import datetime

from collections import defaultdict

DIR_TO_EDGE = np.array([
    [ 0,-1], # north
    [ 1,-1], # north-east
    [ 1, 0], # east
    [ 1, 1], # south-east
    [ 0, 1], # south
    [-1, 1], # south-west
    [-1, 0], # west
    [-1,-1], # north-west
    [ 0, 0], # self-loop
])

PI = tf.constant(np.pi)

QUADRANT_IDS = [
        [0, 1, 2], # N, NE, E
        [2, 3, 4], # E, SE, S
        [4, 5, 6], # S, SW, W
        [6, 7, 0], # W, NW, N
        [8, 8, 8], # SELF-LOOP
    ]

DIRECTIONAL_IDS = tf.range(9)[:,tf.newaxis]

def unstack_time(x, n_timesteps, axis=1):
    '''
    Splits the feature dimension
    into n_timesteps and stack it into a new axis.
    if axis=0: [..., features] -> [time, ..., features // n_timesteps]
    '''
    x = tf.split(x, n_timesteps, axis=-1)
    x = tf.stack(x, axis=axis)
    return x

# ------------ TensorFlow Feature Engineering ----------------


def sample_quadrant_subgraphs(edge_index, direction):
    # as one edge may be in 2 partitions, those edges need to be repeated
    REP_IDS = [0, 2, 4, 6]
    direction = tf.cast(direction, tf.int32)
    direction = direction[:, tf.newaxis]
    edge_repeats = (1 - tf.minimum(1, tf.reduce_min(tf.abs(direction - REP_IDS), axis=-1))) + 1
    edge_index = tf.repeat(edge_index, edge_repeats, axis=0)
    direction = direction[:, tf.newaxis] - QUADRANT_IDS
    subgraph_ids = tf.where(tf.reduce_min(tf.abs(direction), axis=-1) == 0)[:, 1]
    subgraph_ids = tf.cast(subgraph_ids, tf.int32)
    subgraphs = tf.dynamic_partition(edge_index, subgraph_ids, 5)
    return tuple(subgraphs[:4])

def sin_cos_encoding(xs, max_x):
    '''
    Generate sin-cos encoding: sin(2pi * xs/max_x) || cos(2pi * xs/max_x)
    '''
    ts = tf.cast(xs, tf.float32) / max_x * 2 * PI
    return tf.stack((tf.math.sin(ts), tf.math.cos(ts)), axis=-1)

# ------------ TensorFlow Preprocessing methods --------------

def preprocess_pixel_values(traffic_data, scale):
    '''
    Preprosessing for the pixel values.
    Performs casting to float32 and normalization
    to the interval [0, 1].
    '''
    # traffic_data: [time, height, width, features]
    # cast to float32
    traffic_data = tf.cast(traffic_data, tf.float32)
    # scale to [0, 1]
    traffic_data = traffic_data / scale
    return traffic_data

# ------------ TensorFlow File parsing methods --------------

def mask_invalid_ids(graph_data):
    # mask all northbound connections in the first row
    graph_data[[0, 1,-1],  0] *= 0
    # mask all eastbound connections in the last column
    graph_data[[1, 2, 3], :, -1] *= 0
    # mask all southbound connections in the last row
    graph_data[[3, 4, 5], -1] *= 0
    # mask all westbound connections in the first column
    graph_data[[5, 6, 7], :,  0] *= 0
    return graph_data

def image_to_graph(graph_data, return_img_mask=False, add_selfloops=True):
    '''
    Converts image to graph.
    Parameters
        graph_data : [8, height, width] 
            8 binary feature maps of size height x width.
            The feature maps encode edges in eight directions 
            on the pixel grid. (N, NE, E, SE, S, SW, W, NW).
        return_img_mask : bool
            If True, returns a mask to mask all non-node pixels in the images.
    Returns 
        node position: [n, 2]
            Returns the positions on the x-y-grid [n, 2]
        edge index of the graph, i.e. 
            edge_index: [e, 2] where each edge e_i is defined by a tuple
            containing the two indices of pairwise connected nodes.
        img_mask : [height, width]
            A mask that contains a 1 if that pixel is a node, 0 otherwise.
    '''
    m_dims, n_dims = graph_data.shape[1:]
    graph_data = mask_invalid_ids(graph_data)
    graph_data = graph_data / np.max(graph_data)
    flat_map = graph_data.reshape(8, -1)

    node_map = np.max(flat_map, axis=0).astype(np.int32)
    node_mask = node_map == 1

    if add_selfloops:
        flat_map = np.concatenate((flat_map, node_map[np.newaxis]), axis=0)

    n_nodes = np.sum(node_map)
    node_ids = np.arange(n_nodes)
    nodes = np.where(node_map)[0]
    node_map[node_mask] = node_ids
    
    # transpose flat_map to keep the node ids in order
    # tf.where would otherwise sort by direction
    node_pos, direction = np.where(flat_map.T)
    
    relative_edge_id = DIR_TO_EDGE[direction] * [1, n_dims]
    relative_edge_id = np.sum(relative_edge_id, axis=-1)
    
    src_node = node_map[node_pos]
    target_node = node_map[node_pos + relative_edge_id]

    m = nodes // n_dims
    n = nodes % n_dims

    node_position = np.stack((m, n), axis=-1)
    edge_index = np.stack((src_node, target_node), axis=-1)
    
    if return_img_mask:
        node_mask = node_mask.reshape(graph_data.shape[1:])
        return node_position, edge_index, direction, node_mask
        
    return node_position, edge_index, direction

def graph_to_image(nodes, img_mask):
    '''
    This function could use scatter_nd, but gathering the node values
    is empirically faster.
    '''
    mask_shape = tf.shape(img_mask)
    mask_ndims = img_mask.shape.ndims

    if mask_ndims == 2 or mask_ndims == 3:
        mask_flat = tf.reshape(img_mask, -1)
    else:
        raise ValueError(f'Shape incompatible dimensions (shape = {img_mask.shape}; '
                         f'ndim = {mask_shape.ndim}). Expected is ndim = 2 or ndim = 3.')

    mask_idx = tf.cumsum(tf.cast(mask_flat, tf.int32), exclusive=True, axis=-1)
    mask_idx = tf.reshape(mask_idx, mask_shape)
    nodes = tf.gather(nodes, mask_idx)
    feature_dims = nodes.shape.ndims - mask_ndims
    m_shape =  tf.concat((mask_shape, tf.ones(feature_dims, dtype=tf.int32)), axis=0)
    img_mask = tf.reshape(img_mask, m_shape)
    img_mask = tf.cast(img_mask, nodes.dtype)
    nodes = nodes * img_mask

    return nodes

def file_to_city_date(file_path):
    '''
    Extracts the city name and date from a file path
    of the training set (i.e. 8ch.h5-files)
    '''
    file_name = tf.strings.split(file_path, '/')
    file_name = tf.strings.split(file_name, '_')
    file_name = tf.map_fn(lambda x: x[-1], file_name, fn_output_signature=tf.string)
    city = file_name[:,-2]
    date = file_name[:, 0]
    return city, date
    
# ---------- Methods to setup the dataset ----------------

def split_seed_target(traffic_data, seed_len=12):
    n_steps = tf.shape(traffic_data)[0]
    n_seed = seed_len
    n_target = n_steps - n_seed
    seed_nodes, target_nodes = tf.split(traffic_data, [n_seed, n_target], axis=0)
    return seed_nodes, target_nodes
    

def h5_to_consecutive_windows(dataset, sample_size, use_caching=False):
    '''
    Reads the h5 files one by one and windows the results with a shift of 1.
    This option is fast but makes subsequent shuffling very memory intense.
    '''
    @tf.autograph.experimental.do_not_convert
    def interleave_h5(city_file):
        traffic_file, city = tf.unstack(city_file)
        
        def read_h5(traffic_file, city):
            city, date = file_to_city_date([traffic_file])
            weekday = np.int32(datetime(*np.array(date.split('-'), dtype=np.int32)).weekday())

            with h5py.File(traffic_file, 'r') as traffic_data:
                #timesteps = len(traffic_data['array'])
                #lst_out = np.array(traffic_data['array'][:sample_size-1])

                for tidx, t in enumerate(traffic_data['array']):
                    # include the file in the return
                    # to later filter out window entries comping 
                    # from two different files
                    #with h5py.File(traffic_file, 'r') as traffic_data:
                    #    traffic = np.concatenate((lst_out, traffic_data['array'][t:t+1]), axis=0)
                    yield t, city, date, traffic_file, tidx, weekday
                    #lst_out = traffic[1:]
            
        return tf.data.Dataset.from_generator(
            read_h5,
            (tf.uint8, tf.string, tf.string, tf.string, tf.int32, tf.int32),
            args=(traffic_file, city)
        )

    dataset = dataset.interleave(interleave_h5,
                cycle_length=1, block_length=sample_size,
                num_parallel_calls=1)

    if use_caching:
        dataset = dataset.cache()

    dataset = dataset.window(sample_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda traffic, city, date, traffic_file, tidx: 
        tf.data.Dataset.zip((traffic, city, date, traffic_file, tidx))
            .batch(sample_size, drop_remainder=True)
            # filter out time series containing steps from different files
            # t = traffic data; sf = static file; f = traffic file
            .filter(lambda t, c, d, f, tid: tf.reduce_all(tf.equal(f, f[0]))))

    # select only the first of static and traffic files as these are the 
    # same for all timesteps. Convert traffic file to city name and date
    dataset = dataset.map(lambda t, c, d, f, tid: {
                'image'    : preprocess_pixel_values(t, scale=scale),
                'city'     : c,
                'date'     : d,
                'time_idx' : tid }, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def h5_to_parallel_batches(dataset, sample_size, cycle_length=5, use_caching=False, meta=None, scale=None, time_ids=None, meta_files_dir=None):
    '''
    Reads the h5 files in parallel and batches into chunks of length sample_size.
    This option is slower than h5_to_consecutive_windows but leads to a well shuffled dataset.
    This function utilizes a small cache that holds the samples processed in the step before
    and reuses the values whenever possible.

    A bigger cycle_length leads to better shuffling, 
        - maximum value for cycle_length is the number of h5 files.
    '''
    @tf.autograph.experimental.do_not_convert
    def read_h5(traffic_file, city, order):
        
        def read_h5_np(traffic_file, city, order):
            traffic_file = traffic_file.decode()
            
            # if the file is from the test set, it must be treated differently
            if 'test' in traffic_file:
                # load meta information
                s_part = traffic_file.split('_')
                s_part = s_part[:-1] + ['additional'] + s_part[-1:]
                meta_file = '_'.join(s_part)
                meta_file = os.path.join(meta_files_dir, city.decode(), os.path.basename(meta_file))
                t = order
                
                with h5py.File(meta_file, 'r') as traffic_data:
                    weekday, time_idx = np.array(traffic_data['array'][t])
                    date = 'unknown'
                    time_idx = np.int32(time_idx)

                with h5py.File(traffic_file, 'r') as traffic_data:
                    #print(list(traffic_data.keys()))
                    next_out = np.array(traffic_data['array'][t])
                    
                t = time_idx
            else:
                date = traffic_file.split('/')[-1].split('_')[0]
                weekday = datetime(*np.array(date.split('-'), dtype=np.int32)).weekday()
                t = order

                with h5py.File(traffic_file, 'r') as traffic_data:
                    next_out = np.array(traffic_data['array'][t:t+sample_size])
                    
            weekday = np.int32(weekday)
            return next_out, city, date, t, weekday

        next_out, city, date, t, weekday = tf.numpy_function(read_h5_np, (traffic_file, city, order), 
            (tf.uint8, tf.string, tf.string, tf.int32, tf.int32))

        return {
            'image'    : preprocess_pixel_values(next_out, scale=scale),
            'city'     : city,
            'date'     : date,
            'time_idx' : t,
            'weekday'  : weekday
        }

    def build_random_index(city_file):
        def build_np(city_file):
            traffic_file, city = city_file
            _, date = file_to_city_date([traffic_file])
            
            if meta is None or 'test' in traffic_file.decode():
                with h5py.File(traffic_file, 'r') as traffic_data:
                    timesteps = len(traffic_data['array'])
            else:
                timesteps = meta[city.decode()]['timesteps'][date.numpy()[0].decode()]
            
            if 'test' in traffic_file.decode():
                n_steps = timesteps
            else:
                n_steps = timesteps-(sample_size-1)
            
            if time_ids is not None:
                assert max(time_ids) < n_steps
                order = np.array(time_ids, dtype=np.int32)
                n_steps = len(order)
            else:
                order = np.arange(n_steps, dtype=np.int32)
                np.random.shuffle(order)
            
            traffic_file = tf.repeat([traffic_file], n_steps, axis=0)
            city = tf.repeat([city], n_steps, axis=0)

            return traffic_file, city, order

        traffic_file, city, order = tf.numpy_function(build_np, (city_file, ), 
            (tf.string, tf.string, tf.int32))
        return tf.data.Dataset.from_tensor_slices((traffic_file, city, order))

    dataset = dataset.shuffle(cycle_length)
    dataset = dataset.flat_map(build_random_index)
    dataset = dataset.shuffle(cycle_length * 12 * 24)
    dataset = dataset.map(read_h5, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset