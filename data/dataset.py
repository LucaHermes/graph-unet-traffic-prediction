from collections import namedtuple, defaultdict
from pathlib import Path
import numpy as np
import h5py
import warnings
import os
import re
from multiprocessing import Pool
import tensorflow as tf

import data.data_utils as data_utils


class T4CDataset:

    def __init__(self, data_dir, cities=None, out_dir=None, include_pattern=None, exclude_pattern=None, 
        dynamic_files_suffix='_8ch.h5', static_files_suffix='_static.h5', flipped=False):
        '''
        Dataset class for managing the T4C data. If you want to use a tensorflow
        data pipeline, use T4CDatasetTF. 

        data_dir: 
            path to the data folder
        cities: 
            only use the given cities, defaults to all cities
            e.g. ['BARCELONA', 'BANGKOK']
        out_dir:
            the directory to store processed graphs, 
            defaults to /tmp/t4c_data
        '''
        self.data_dir = data_dir
        self.cities = cities
        self.exclude_pattern = exclude_pattern
        self.include_pattern = include_pattern
        self.static_files_suffix = static_files_suffix
        self.dynamic_files_suffix = dynamic_files_suffix
        self.data_dir_subdirs = len(Path(data_dir).parts)
        self.flip_data = flipped

        if include_pattern is None:
            self.include_pattern = ''
        
        if cities is None:
            all_traffic_files = self.get_files(data_dir, '*%s' % dynamic_files_suffix,
                include_pattern=self.include_pattern, exclude_pattern=self.exclude_pattern)
            all_cities = set(map(lambda x: x.parts[self.data_dir_subdirs], all_traffic_files))
            all_static_files = self.get_files(data_dir, '*%s' % static_files_suffix)
            all_static_cities = set(map(lambda x: x.name.split('_')[0], all_static_files))
            self.cities = all_cities.intersection(all_static_cities)
        
        if self.flip_data:
            self.maybe_flip_files()

        self.city_index = dict(zip(self.cities, range(len(self.cities))))
        if out_dir is None:
            out_dir = '/tmp/t4c_data'
        self.out_dir = out_dir
        self.graphs_path = os.path.join(self.out_dir, 'graphs')
            
        self._load()
        self._cache_city_static()
        self._get_city_meta()
        self._preprocess_graphs()
    
    @property
    def total_timesteps(self):
        return sum([ sum(v['timesteps'].values()) for v in self.meta.values() ])

    def get_files(self, directory, pattern, include_pattern=None, exclude_pattern=None):
        files = Path(directory).rglob(pattern)
        # only include files matching the include pattern
        if include_pattern:
            files = filter(lambda f: re.match(include_pattern, str(f)), files)
        # exclude files matching the exclude pattern
        if exclude_pattern:
            files = filter(lambda f: not re.match(exclude_pattern, str(f)), files)
            
        return list(files)
    
    def init_random_state(self, seed=42):
        self.rnd_gen = np.random.RandomState(seed=seed)
        
    def _get_city_meta(self):
        self.meta = {}
        meta_data = ['timesteps', 'height', 'width', 'channels']
        
        for city in self.cities:
            self.meta[city] = {}
            self.meta[city]['timesteps'] = {}
            self.meta[city]['total_timesteps'] = 0
            
            for file in self.dyn_files[city]:
                with h5py.File(file, 'r') as f:
                    data_shape = f['array'].shape
                
                timesteps = data_shape[0]
                date = file.split('/')[-1].split('_')[0]
                self.meta[city]['timesteps'][date] = timesteps
                self.meta[city]['total_timesteps'] += timesteps
            
            self.meta[city]['height'] = data_shape[-3]
            self.meta[city]['width'] = data_shape[-2]
            self.meta[city]['channels'] = data_shape[-1]
    
    def _load(self):
        '''
        Searches self.data_dir for the dataset files and
        stores them.
        '''
        self.static_files = {}
        self.dyn_files = {}
        
        for city in self.cities:
            # load static data
            city_static = self.get_files(self.data_dir, f'*{city}{self.static_files_suffix}')
            city_static = map(lambda p: str(p), city_static)
            static_file = next(city_static, None)

            if not static_file: 
                continue

            self.static_files[city] = static_file
            
            # load dynamic data
            city_dyn = self.get_files(self.data_dir, f'*{city}*{self.dynamic_files_suffix}',
                include_pattern=self.include_pattern, exclude_pattern=self.exclude_pattern)
            city_dyn = map(lambda p: str(p), city_dyn)
            self.dyn_files[city] = list(city_dyn)
        
    def maybe_flip_files(self):
        '''
        Flips the dataset files horizontally and then vertically.
        '''
        flip_dynamic_files_suffix = '_dyn_flipped.h5'
        flip_static_files_suffix = '_static_flipped.h5'
        
        for city in self.cities:
            # maybe flip static files
            city_static = self.get_files(self.data_dir, f'*{city}{flip_static_files_suffix}')
            
            if len(city_static) == 0:
                # flip static files
                city_static = self.get_files(self.data_dir, f'*{city}{self.static_files_suffix}')
                
                for f in city_static:
                    filename = os.path.basename(f)
                    filename = filename[:-len(self.static_files_suffix)] + flip_static_files_suffix
                    out_file = os.path.join(self.data_dir, city, filename)
                    
                    with h5py.File(out_file, 'w') as wf:
                        with h5py.File(f, 'r') as rf:
                            data = np.array(rf['array'])
                            # flip spacial
                            data = data[:, ::-1, ::-1]
                            # flip connections
                            data = data[[0, 5, 6, 7, 8, 1, 2, 3, 4]]
                        wf.create_dataset('array', data=data, 
                            chunks=data.shape, 
                            dtype="uint8", compression="lzf")
                        
            # maybe flip dynamic files
            city_dyn = self.get_files(self.data_dir, f'*{city}*{flip_dynamic_files_suffix}',
                include_pattern=self.include_pattern, exclude_pattern=self.exclude_pattern)
            
            if len(city_dyn) == 0:
                # flip static files
                city_dyn = self.get_files(self.data_dir, f'*{city}*{self.dynamic_files_suffix}',
                    include_pattern=self.include_pattern, exclude_pattern=self.exclude_pattern)
                
                for f in city_dyn:
                    filename = os.path.basename(f)
                    filename = filename[:-len(self.dynamic_files_suffix)] + flip_dynamic_files_suffix
                    dirname = os.path.dirname(f)
                    out_file = os.path.join(dirname, filename)
                    with h5py.File(out_file, 'w') as wf:
                        with h5py.File(f, 'r') as rf:
                            data = np.array(rf['array'])
                            data = data[:,::-1,::-1,[4,5,6,7,0,1,2,3]]
                        wf.create_dataset('array', data=data, 
                            chunks=(1, *data.shape[1:]), 
                            dtype="uint8", compression="lzf")
        
        self.static_files_suffix = flip_static_files_suffix
        self.dynamic_files_suffix = flip_dynamic_files_suffix
        

    def _cache_city_static(self):
        '''
        Loads the static data of the cities in memory.
        These are all separate lists to acess them via tensorflow.
        '''
        self.city_street_map = []
        self.city_graph_map = []
        self.city_edges = []
        self.city_edge_dir = []
        self.city_img_mask = []
        self.city_node_loc = []
        self.city_node_count = []
        self.city_edge_count = []

        for city in self.cities:
            print(f'Caching city {city} ...')
            with h5py.File(self.static_files[city], 'r') as f:
                data = np.array(f['array'])
                
                street_map = data[0]
                graph_maps = data[1:]
                node_loc, edge_index, direction, img_mask = data_utils.image_to_graph(
                    graph_maps, 
                    return_img_mask=True)

            self.city_street_map.append(street_map)
            self.city_graph_map.append(graph_maps)
            self.city_edges.append(edge_index)
            self.city_edge_dir.append(direction)
            self.city_img_mask.append(img_mask)
            self.city_node_loc.append(node_loc)
            self.city_node_count.append(img_mask.sum())
            self.city_edge_count.append(len(edge_index))

        self.max_nodes = max(self.city_node_count)
        self.max_edges = max(self.city_edge_count)
    
    def _preprocess_traffic_file(self, city_file):
        '''
        Loads the data in city_file, converts in into graph format and
        saves the graph data in a new file in self.out_dir
        '''
        city, file = city_file
        filename = os.path.basename(file)
        out_file = os.path.join(self.graphs_path, filename)

        if os.path.exists(out_file):
            return city, out_file

        city_idx = self.city_index[city]
        city_mask = self.city_img_mask[city_idx]
        os.makedirs(self.graphs_path, exist_ok=True)

        with h5py.File(out_file, 'w') as f:
            with h5py.File(file, 'r') as rf:
                data = np.array(rf['array'])
                # test data is different from training data here, as it provides distinct
                # time slots
                if len(data.shape) == 4:
                    traffic_nodes = data[:,city_mask]
                    
                elif len(data.shape) == 5:
                    traffic_nodes = data[:,:,city_mask]
                f.create_dataset('array', data=traffic_nodes, 
                    chunks=(1, *traffic_nodes.shape[1:]), 
                    dtype="uint8", compression="lzf")
        
        print(f'Graph created for {city} with shape {traffic_nodes.shape}')

        return city, out_file

    def _preprocess_graphs(self):
        '''
        Loads the data of all cities, converts in into graph format and
        saves the graph data in a new file in self.out_dir
        '''
        city_files = [ (np.repeat(c, len(fs), axis=0), np.array(fs))
                       for c, fs in self.dyn_files.items() ]

        city_files = np.concatenate(city_files, -1)
        city_files = list(zip(*city_files))
        self.graph_files = defaultdict(list)
        chunksize = max(len(city_files) // os.cpu_count(), 1)

        with Pool(processes=None) as pool:
            for c, f in pool.imap_unordered(self._preprocess_traffic_file, city_files, chunksize=chunksize):
                print('Starting preprocessing task ...')
                self.graph_files[c].append(f)
                print(f)
        
        self.graph_files = dict(self.graph_files)

class T4CDatasetTF(T4CDataset):

    def __init__(self, data_dir, cities=None, out_dir=None, prefetch_cache=None, data_scale=255, seed=42, 
                 timesteps=None, **kwargs):
        '''
        Dataset class for loading the T4C data as a tensorflow data pipeline.
        Can be used to output graph data or image data:

        Example for graphs:

            t4c = data.dataset.T4CDatasetTF(data_dir, cities)
            t4c = t4c.as_graphs(seed_len, target_len)

            One iteration outputs a dictionary with keys 'graph', 'city', 'date',
            where graph is a dictionary with keys: 'nodes', 'edge_index', 'edges'.
            For batched graphs use as_batched_graphs instead of as_graphs.

        Example for images:

            t4c = data.dataset.T4CDatasetTF(data_dir, ['BARCELONA', 'BANGKOK'])
            t4c = t4c.as_images(seed_len, target_len)

            one iteration outputs a dictionary with keys 'image', 'city', 'date'.
            Where traffic is a tensor of shape [time, height, width, features]
            For batches: t4c = t4c.batch(batch_size)

        Transforms to the data can be applied via t4c.add_transform(fn) and will
        be execited in that place of the data pipeline:

            t4c = data.dataset.T4CDatasetTF(data_dir, cities)
            t4c = t4c.as_graphs(seed_len, target_len)
            t4c = t4c.add_transform(fn)


        data_dir: 
            path to the data folder.
        cities: 
            only use the given cities, defaults to all cities
            e.g. ['BARCELONA', 'BANGKOK'].
        out_dir:
            the directory to store processed graphs, 
            defaults to /tmp/t4c_data.
        prefetch_cache:
            Defines the number of samples that should be pre loaded during
            the train step.
        data_scale:
            A scalar that is used to scale the pixel values by 1/data_scale.
        '''
        super(T4CDatasetTF, self).__init__(data_dir, cities=cities, out_dir=out_dir, **kwargs)

        # create a tf lookup table to query static city data
        print('Converting arrays to tensors ...')
        self.city_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(self.city_index.keys()), 
                np.array(list(self.city_index.values()), dtype=np.int32)),
            default_value=-1)

        # convert static city data
        self.city_street_map = tf.stack(self.city_street_map)
        self.city_graph_map = tf.stack(self.city_graph_map)
        self.city_edges = tf.ragged.stack(self.city_edges)
        self.city_edge_dir = tf.ragged.stack(self.city_edge_dir)
        self.city_node_loc = tf.ragged.stack(self.city_node_loc)
        self.city_img_mask = tf.stack(self.city_img_mask)
        self.city_node_count = tf.stack(self.city_node_count)
        self.city_edge_count = tf.stack(self.city_edge_count)
        self.prefetch_cache = prefetch_cache if prefetch_cache is not None else tf.data.AUTOTUNE
        self.timesteps = timesteps
        self.sample_size = None
        self.data_scale = float(data_scale)
        self.out_dir = out_dir
        self.seed = seed
        print('Done')

    @property
    def size(self):
        if self.timesteps is not None:
            n_files = len(np.concatenate(list(self.dyn_files.values())))
            n_timesteps = len(self.timesteps)
            steps =  n_files *n_timesteps
        else:
            steps = 0
            sample_size = self.sample_size
            sample_size = 24 if sample_size is None else sample_size
            time_offset = sample_size - 1
            for v in self.meta.values():
                file_timesteps = np.array(list(v['timesteps'].values()))
                steps += np.sum(file_timesteps - time_offset)
        return steps

    def __iter__(self):
        dataset = self.dataset.prefetch(self.prefetch_cache)
        return iter(dataset)

    def _from_file_dict(self, files, seed_len=12, target_len=12, shuffle=True, use_caching=False):
        '''
        Initializes the data pipeline as follows:
        Data files -> shuffle -> read h5 -> scale pixels

        One iteration outputs a dictionary with keys 'image', 'city', 'date',
        where traffic is a tensor of shape [time, height, width, features].
        For batches: t4c = t4c.batch(batch_size)
        '''
        self.sample_size = seed_len + target_len

        city_files = [ np.stack((v, np.repeat(k, len(v))), -1) 
                  for k, v in files.items() ]
        city_files = np.concatenate(city_files)

        ds = tf.data.Dataset.from_tensor_slices(city_files)
        
        if shuffle:
            # perform perfect shuffling over the days in the dataset
            # each day is stores as one file
            ds = ds.shuffle(len(city_files))
            ds = data_utils.h5_to_parallel_batches(ds, self.sample_size, meta=self.meta, cycle_length=len(city_files), meta_files_dir=self.data_dir,
                                                   use_caching=use_caching, scale=self.data_scale, time_ids=self.timesteps)
        else:
            ds = data_utils.h5_to_consecutive_windows(ds, self.sample_size, use_caching=use_caching, scale=self.data_scale)
            
        self.dataset = ds
        return self

    def get_img_mask(self, city):
        return tf.gather(self.city_img_mask, self.city_lookup.lookup(city))

    def get_street_map(self, city, as_nodes_list=False):
        c_idx = self.city_lookup.lookup(city)
        if as_nodes_list:
            mask = self.get_img_mask(city)
            street_map = tf.gather(self.city_street_map, c_idx)
            return tf.gather_nd(street_map, tf.where(mask))
        return tf.gather(self.city_street_map, c_idx)

    def batch(self, batch_size):
        if self.mode == 'graph':
            raise NotImplementedError('The batch method only works for images.'
                'To get a dataset of batched graphs, use as_batched_graphs().')
        self.dataset = self.dataset.batch(batch_size)
        return self

    def add_transform(self, f):
        '''
        Add a preprocessing function that will be applied in place,
        if f should  be applied prior to conversion to graph, then:
            t4c = T4CDatasetTF().add_transform(f).as_graphs(),
        if f should be applied after:
            t4c = T4CDatasetTF().as_graphs().add_transform(f),
        '''
        self.dataset = self.dataset.map(f, 
            num_parallel_calls=tf.data.AUTOTUNE)
        return self

    def as_graphs(self, seed_len=12, target_len=12, shuffle=True, use_caching=False):
        '''
        Initializes the data pipeline as follows:
        Data files -> shuffle -> read h5 graphs -> scale pixels

        One iteration outputs a dictionary with keys 'graph', 'city', 'date',
        where graph is a dictionary with keys: 'nodes', 'edge_index', 'edges'.
        '''
        sample_size = seed_len + target_len
        self._from_file_dict(self.graph_files, seed_len, target_len, shuffle, use_caching)

        def _to_graph(data):
            city_ids = self.city_lookup.lookup(data['city'])[tf.newaxis]
            city_ids.set_shape(1)
            edge_index = tf.gather(self.city_edges, city_ids)[0].to_tensor()
            mask = tf.gather(self.city_img_mask, city_ids)[0]
            edge_direction = tf.gather(self.city_edge_dir, city_ids)[0]
            seed_nodes, target_nodes = data_utils.split_seed_target(data['image'], seed_len)
            return {   
                'graph' : {
                    'seed_nodes'     : seed_nodes,
                    'target_nodes'   : target_nodes,
                    'edge_index'     : edge_index,
                    'edges'          : tf.ones(tf.shape(edge_index)[0]),
                    'edge_direction' : edge_direction },
                'city'     : data['city'][tf.newaxis],
                'date'     : data['date'][tf.newaxis],
                'time_idx' : data['time_idx'][tf.newaxis],
                'weekday'  : data['weekday'][tf.newaxis]
            }

        self.dataset = self.dataset.map(_to_graph, num_parallel_calls=tf.data.AUTOTUNE)
        self.mode = 'graph'
        return self

    def as_images(self, seed_len=12, target_len=12, shuffle=True, use_caching=False, concat_time=True):
        '''
        Initializes the data pipeline as follows:
        Data files -> shuffle -> read h5 images -> scale pixels

        One iteration outputs a dictionary with keys 'image', 'city', 'date',
        where traffic is a tensor of shape [time, height, width, features].
        For batches: t4c = t4c.batch(batch_size)
        '''
        self.mode = 'images'

        self._from_file_dict(self.dyn_files, seed_len, target_len, shuffle, use_caching)

        def add_features(data):
            city_idx = self.city_lookup.lookup(data['city'])[tf.newaxis]
            city_street_map = tf.gather(self.city_street_map, city_idx)
            city_graph_map = tf.gather(self.city_graph_map, city_idx)[0]
            static_data = tf.concat((city_street_map, city_graph_map), axis=0)
            static_data = tf.transpose(static_data, [1,2,0])
            seed_target_split = data_utils.split_seed_target(data['image'], seed_len)
            return {
                'image'    : dict(zip(
                    ['seed_image', 'target_image', 'static_data'], 
                    seed_target_split + (static_data,))),
                'city'     : data['city'], 
                'date'     : data['date'],
                'time_idx' : data['time_idx'],
                'weekday'  : data['weekday']
            }

        self.dataset = self.dataset.map(add_features, num_parallel_calls=tf.data.AUTOTUNE)
        return self

    def concat_time(self, x):
        '''
        Concatenates the time axis into the feature axis:
        if in graph mode:
        [time, nodes, features] -> [nodes, features * time]
        if in image mode:
        [..., time, height, width, features] -> [..., height, width, features * time]
        '''
        if self.mode == 'graph':
            for key in ['seed_nodes', 'target_nodes']:
                nodes = x['graph'][key]
                nodes = tf.transpose(nodes, [1, 0, 2])
                n_nodes = tf.shape(nodes)[0]
                x['graph'][key] = tf.reshape(nodes, [n_nodes, -1])

        if self.mode == 'images':
            for key in ['seed_image', 'target_image']:
                img = x['image'][key]
                img = tf.transpose(img, [0, 2, 3, 1, 4])
                img_shape = tf.concat((tf.shape(img)[:-2], [-1]), axis=0)
                img = tf.reshape(img, img_shape)
                x['image'][key] = img

        return x
