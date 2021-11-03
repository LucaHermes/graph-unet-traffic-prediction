import h5py
import re
import os
from datetime import datetime
from pathlib import Path
# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import zipfile
import tensorflow as tf
from train import create_cli_parser, init_model
from models.base_model import TSPEC_D
from train_utils import build_data_pipeline
from config import DEFAULTS
import data
import data.data_utils as data_utils


# input signature for the evaluation datasets is different, as no ground truth is given
TSPEC_D_test = TSPEC_D.copy()
TSPEC_D_test[0]['graph']['target_nodes'] = tf.TensorSpec(shape=(None, 0), dtype=tf.float32, name=None)
TSPEC_D_test.append(tf.TensorSpec(shape=(495, 436), dtype=tf.bool, name=None))
TSPEC_D_test.append(tf.TensorSpec(shape=None, dtype=tf.float32, name=None))


def package_submissions(test_set, submissions, model_id, model_name, challenge_name, out_dir='submissions'):
    date_time = datetime.now()
    time_str = date_time.strftime('%Y%m%d%H%M')
    out_dir = os.path.join(out_dir, model_id, date_time.isoformat())
    zip_name = 'submission_%s_%s_%s.zip' % (model_name, challenge_name, time_str)
    zip_file = os.path.join(out_dir, zip_name)
    
    with zipfile.ZipFile(zip_file, 'w') as z:
        for city, files in test_set.dyn_files.items():
            city_path = os.path.join(out_dir, city)
            
            if not os.path.exists(city_path):
                os.makedirs(city_path)
                
            f_part = files[0].split('_')
            meta_file = f_part[:-1] + ['additional'] + f_part[-1:]
            meta_file = '_'.join(meta_file)
            
            data_file = test_set.dyn_files[city][0]
            file_name = os.path.basename(data_file) + '.h5'
            out_file = os.path.join(city_path, file_name)
        
            with h5py.File(out_file, 'a') as f:
                with h5py.File(data_file, 'r') as tmp:
                    n_items = tmp['array'].shape[0]
                    print(city, n_items, data_file)
                    compression = tmp['array'].compression
                    shape = next(iter(submissions.values())).shape
                f.create_dataset('array', shape=(n_items, *shape), chunks=(1, *shape), 
                                 compression=compression)

                with h5py.File(meta_file, 'r') as meta_f:
                    for i, (weekday, tidx) in enumerate(meta_f['array']):
                        f['array'][i] = submissions[(city.encode(), weekday, tidx)]
            z.write(out_file)
        print(f'[Submission] Zip-file created {zip_file}')

def graph_to_image(nodes, img_mask):
    mask_shape = tf.shape(img_mask)
    mask_idx = tf.cast(tf.where(img_mask), tf.int32)
    nodes = tf.scatter_nd(mask_idx, nodes, shape=tf.concat((mask_shape, tf.shape(nodes)[1:]), axis=0))
    return nodes

@tf.function(input_signature=TSPEC_D_test)
def predict_sample(sample, img_mask, data_scale=255):
    pred = model._call(sample)
    pred_img = graph_to_image(pred, img_mask)
    pred_img = tf.stack(tf.split(pred_img, 12, axis=-1))
    # get 5min, 10min and 15min, 30min, 45min and 60min predictions
    #              0  1  2        5        8       11   [sample-idx]
    # input steps: 5 10 15 20 25 30 35 40 45 50 55 60   [min]
    pred_img = tf.gather(pred_img, [0, 1, 2, 5, 8, 11])
    pred_img = tf.cast(tf.round(pred_img * data_scale), tf.uint8)
    pred_img = tf.clip_by_value(pred_img, 0, tf.cast(data_scale, tf.uint8))
    return pred_img

def create_submission(model, config, challenge):
    test_set = data.dataset.T4CDatasetTF(
        config['data_dir'], 
        config['include_cities'],
        include_pattern=r'.*test_%s.*' % challenge, 
        dynamic_files_suffix='.h5')
    test_set = build_data_pipeline(test_set, config)
    
    model_input = []
    submission = {}
    
    for i, sample in enumerate(test_set):
        model_input.append(sample)

        tidx = sample['time_idx'][0].numpy()
        weekday = sample['weekday'][0].numpy()
        city = sample['city'][0]
        
        img_mask = test_set.get_img_mask(city)
        pred_img = predict_sample(sample, img_mask, test_set.data_scale)
        pred_img = pred_img.numpy()

        city = city.numpy()
        submission[(city, weekday, tidx)] = pred_img
        print(f'Packaging challenge "{challenge}", sample', i, end='\r')
    
    package_submissions(test_set, submission, model.id, model.architecture, challenge)


if __name__ == '__main__':
    parser = create_cli_parser()
    args = parser.parse_args()
    config = DEFAULTS
    config.update(vars(args))
    
    model, op = init_model(config['model'][0], config)
    print(f'[Submission] Creating submission for model {config["model"][0]}')
    print(f'[Submission]    * Using checkpoint: {config["checkpoint"]}')
    print(f'[Submission]    * Model step:       {model.global_step.numpy()}')
    create_submission(model, config, 'temporal')
    create_submission(model, config, 'spatiotemporal')