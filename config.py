import tensorflow as tf

from models import graph_baselines
from models import unet
from models import graph_unet
import train_utils

# defaults for the arguments of the cli
DEFAULTS = {
    # default model hyperparameters
    # The number of units for the first (and last) U-Net block
    'units'               : 32,
    # The number of output neurons (default 96)
    'out_units'           : 12*8,
    # The number of downsampling blocks and upsampling blocks
    'depth'               : 3,
    # The layer type used in the Graph-UNet model (s. layers/__init__.py)
    'layer_type'          : 'geo_quadrant_gcn',
    'activation'          : 'relu',
    'use_bias'            : True,
    # If True, includes a global node in the graph layers (default)
    'use_global'          : True,
    # If True, includes a global node in the graph layers (default)
    'output_activation'   : None, #'relu',

    # default training parameters
    'data_dir'            : 'data/raw/',
    # type of data, either "image" for convolution-based models, or "graph" for graph-based models
    'data_type'           : 'image',
    # group name (used by the wandb logger to group runs)
    'group'               : 'Baselines',
    'batch'               : 1,
    'epochs'              : 15,
    'learning_rate'       : 'warmup+expDecay', #1e-3,
    'optimizer'           : 'adam',
    'loss'                : 'mse',
    # as the data is quite large, we use gradient accumulation over the specified number of steps
    'acc_gradients_steps' : 16,
    'add_temp_encoding'   : True,
    'add_street_encoding' : False,
    'validation_fraction' : 0.1,
    'seed_len'            : 12,
    'target_len'          : 12,
    'ckpts_dir'           : './ckpts',
}

# CLI argument --model 
# list of all possible models
MODELS = {
    'UNet'       : unet.VanillaUNet,
    'GraphUNet'  : graph_unet.GraphUNet,
}
LOSSES = {
    # option for training all models
    'mse'     : tf.keras.losses.MSE,
    'mae'     : tf.keras.losses.MAE
}

LEARNING_SCHEDULES = {
	'warmup+expDecay' : lambda model: lambda: train_utils.warmupExpDecay(model.global_step)
}