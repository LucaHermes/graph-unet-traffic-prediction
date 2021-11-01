import os
import argparse
# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from datetime import datetime
import wandb

# enable GPU growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

import train_utils
import data
from config import DEFAULTS, MODELS, LEARNING_SCHEDULES, LOSSES

# for possible training- and model-configs, see config.py

def init_datasets(config, train_pattern=None, val_pattern=r'.*2019-03-31.*', 
    val_pattern_II=r'.*2019-03-20_ISTANBUL.*', seed=2):
    train_set = data.dataset.T4CDatasetTF(config['data_dir'], config['include_cities'], 
        exclude_pattern=r'(%s|%s)' % (val_pattern, val_pattern_II), include_pattern=train_pattern)
    val_set = data.dataset.T4CDatasetTF(config['data_dir'], config['include_cities'], 
        include_pattern=val_pattern, exclude_pattern=train_pattern)
    
    
    ts = np.random.RandomState(seed=seed).randint(0, 288-32, 16)
    val_set_II = data.dataset.T4CDatasetTF(config['data_dir'], config['include_cities'], 
        include_pattern=val_pattern_II, timesteps=ts)
    val_set_II_flipped = data.dataset.T4CDatasetTF(config['data_dir'], config['include_cities'], 
        include_pattern=val_pattern_II, flipped=True, timesteps=ts)

    for t4c in [train_set, val_set, val_set_II, val_set_II_flipped]:
        t4c = train_utils.build_data_pipeline(t4c, config)

    return train_set, val_set, val_set_II, val_set_II_flipped

def init_model(model_name, config):
    Model = MODELS[model_name]
    model = Model(**config)

    if isinstance(config['learning_rate'], float):
        learning_rate = lambda: config['learning_rate']
    else:
        learning_rate = LEARNING_SCHEDULES[config['learning_rate']](model)

    optimizer = tf.keras.optimizers.get({
        'class_name' : config['optimizer'],
        'config'     : { 
            'learning_rate' : learning_rate
        }
    })

    if config['checkpoint'] is not None:
        model.load(config['checkpoint'], optimizer=optimizer)

    if config['finetuning']:
        date = datetime.now().strftime('%d-%m_%H-%M-%S')
        model._id.assign('%s_finetune' % model.id)

    return model, optimizer

def init_logger(model, config):
    model_name = model.architecture

    if hasattr(model, 'layer_type'):
        model_name
        model_name = '%s (%s)' % (model_name, model.layer_type)

    logger_config = {
        'project' : 'Traffic4Cast_FullSize',
        'group'   : config['group'],
        'name'    : model_name, #'GResNet (%s)' % model_config['layer_type'],
        'config'  : config,
        'id'      : model.id
    }

    if config['checkpoint'] is not None:
        logger_config['resume'] = 'allow'
        print('> Resuming wandb-run with id %s' % model.id)

    wandb.login()
    wandb.init(**logger_config)
    
def train(config):

    train_set, val_set, val_sm, val_sm_flip = init_datasets(config)
    
    loss_fn = LOSSES[config['loss']]

    for m in config['model']:
        model, optimizer = init_model(m, config)
        init_logger(model, config)
            
        model.train(
            train_set,
            optimizer, 
            config['epochs'], 
            loss_fn=loss_fn, 
            ckpts_dir=config['ckpts_dir'],
            validation_dataset=val_set,
            validation_set_sm=val_sm,
            validation_set_sm_spatial=val_sm_flip,
            acc_gradients_steps=config['acc_gradients_steps'])

def create_cli_parser():
    parser = argparse.ArgumentParser(add_help=True,
        description='Train models on the Traffic4Cast dataset.')

    # ------------------------- Training setup args -------------------------
    parser.add_argument('--model', '-m', type=str, nargs='+', required=True,
                    help='The model to train, one of "%s". If "all" is passed, all models '
                         'will be trained.' % '", "'.join(list(MODELS.keys())))
    parser.add_argument('--depth', '-D', type=int, nargs='?', default=DEFAULTS['depth'],
                    help='The number of layers/blocks, defaults to "%s".' % DEFAULTS['epochs'])
    parser.add_argument('--units', '-U', type=int, nargs='?', default=DEFAULTS['units'],
                    help='The number of units used by the models, defaults to "%s".' % DEFAULTS['units'])
    parser.add_argument('--out_units', '-O', type=int, nargs='?', default=DEFAULTS['out_units'],
                    help='The number of output units used by the models, defaults to "%s".' % DEFAULTS['out_units'])
    parser.add_argument('--activation', '-A', type=str, nargs='?', default=DEFAULTS['activation'],
                    help='The layer activation of the model, defaults to "%s".' % DEFAULTS['activation'])
    parser.add_argument('--layer_type', '-L', type=str, nargs='?', default=DEFAULTS['layer_type'],
                    help='Layer type of the model %s.' % DEFAULTS['layer_type'])
    parser.add_argument('--use_bias', '-B', type=bool, nargs='?', default=DEFAULTS['use_bias'],
                    help='Whether to use bias term, defaults to %d.' % DEFAULTS['use_bias'])
    parser.add_argument('--use_global', '-G', type=bool, nargs='?', default=DEFAULTS['use_global'],
                    help='Whether to use a global node in the graph layers, '
                         'defaults to %d.' % DEFAULTS['use_global'])
    parser.add_argument('--epochs', '-e', type=int, nargs='?', default=DEFAULTS['epochs'],
                    help='The number of epochs to train, defaults to "%s".' % DEFAULTS['epochs'])
    parser.add_argument('--batch', '-b', type=int, nargs='?', default=DEFAULTS['batch'],
                    help='Batch size, defaults to %d.' % DEFAULTS['batch'])
    parser.add_argument('--temporal', '-T', action='store_true',
                    help='Flag to apply explicit temporal processing. Instead of concatenating '
                         'the time steps into the feature dimension, explicit temporal processing is applied. '
                         'Requires a model capable of temporal processing.')
    parser.add_argument('--checkpoint', '-C', type=str, nargs='?', 
                    help='Path to a checkpoint to continue training from there.')
    parser.add_argument('--acc_gradients_steps', '-AG', type=int, nargs='?', default=DEFAULTS['acc_gradients_steps'],
                    help='Number of gradients to accumulate before updating the model, defaults to %d.' % DEFAULTS['acc_gradients_steps'])
    
    # ------------------------- Dataset setup args -------------------------
    parser.add_argument('--data_dir', '-d', type=str, nargs='?', default=DEFAULTS['data_dir'],
                    help='Directory of the dataset, defaults to "%s".' % DEFAULTS['data_dir'])
    parser.add_argument('--data_type', '-t', type=str, nargs='?', default=DEFAULTS['data_type'],
                    help='Data type, either "image" or "graph", defaults to "%s".' % DEFAULTS['data_type'])
    parser.add_argument('--include_cities', '-i', type=str, nargs='+',
                    help='The cities to include, defaults to all cities in data_dir.')

    # ------------------------- Logger setup args -------------------------
    parser.add_argument('--group', '-g', type=str, nargs='?', default=DEFAULTS['group'],
                    help='The group of the model, e.g. "Baselines" (for wandb logging), '
                         'defaults to "%s".' % DEFAULTS['group'])
    parser.add_argument('--nologs', action='store_true',
                    help='Disable wandb logs for this run.')
    parser.add_argument('--finetuning', action='store_true',
                    help='Creates a new run on wandb rather than continuing any old run.')
    return parser

if __name__ == '__main__':
    parser = create_cli_parser()

    args = parser.parse_args()
    config = DEFAULTS
    config.update(vars(args))

    if config['nologs']:
        os.environ['WANDB_MODE'] = 'offline'

    print(config)
    train(config)