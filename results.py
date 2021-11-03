import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from train import create_cli_parser, init_model
from train_utils import build_data_pipeline
from config import DEFAULTS
import data

def create_n_days(n_days, starting_date='2019-04-01', delta_days=1):
    starting_date = datetime.datetime(*list(map(int, starting_date.split('-'))))
    delta = datetime.timedelta(days=delta_days)
    days = [ starting_date + delta * n for n in range(n_days) ]
    days = list(map(lambda d: d.strftime('%Y-%m-%d'), days))
    return days

def graph_to_image(nodes, img_mask):
    mask_shape = tf.shape(img_mask)
    mask_idx = tf.cast(tf.where(img_mask), tf.int32)
    nodes = tf.scatter_nd(mask_idx, nodes, shape=(*mask_shape, *tf.shape(nodes)[1:]))
    return nodes

def create_one_city_dataset(config, city='BERLIN', date_pattern=None, timesteps=None):
    '''
    Generates a dataset containing only a single city at the specified date_pattern
    at the given timesteps
    The default args are the ones used in the paper.
    '''
    if timesteps is None:
        # set default timesteps
        timesteps = np.array([0*12, 6*12, 12*12, 18*12])
    if date_pattern is None:
        date_pattern = '2019-03-(18|19|20)'
       
    include_pattern = r'.*%s_%s.*' % (date_pattern, city.upper())
        
    test_set = data.dataset.T4CDatasetTF(config['data_dir'], config['include_cities'],
                                         include_pattern=include_pattern, 
                                         dynamic_files_suffix='8ch.h5', 
                                         timesteps=timesteps)
    return build_data_pipeline(test_set, config)

def create_quantitative_eval_dataset(config, timesteps=None, n_days=30, starting_date=None):
    '''
    Creates two datasets, one with cities used in training and one that contains 
    horizontally and vertically flipped data.
    '''
    if timesteps is None:
        # set default timesteps
        timesteps = np.array([ hour*12 for hour in range(0, 23, 1)])
    if starting_date is None:
        starting_date = '2019-04-01'
        
    n_days = create_n_days(n_days, starting_date)
    days_pattern = '(%s)' % '|'.join(n_days)

    train_test_set = data.dataset.T4CDatasetTF(config['data_dir'], 
                                               config['include_cities'],
                                               include_pattern=r'.*%s.*' % days_pattern, 
                                               timesteps=timesteps)

    train_test_set = build_data_pipeline(train_test_set, config)

    spatial_test_set = data.dataset.T4CDatasetTF(config['data_dir'], 
                                                 config['include_cities'],
                                                 include_pattern=r'.*%s.*' % days_pattern, 
                                                 dynamic_files_suffix='8ch.h5', 
                                                 timesteps=timesteps, 
                                                 flipped=True)

    spatial_test_set = build_data_pipeline(spatial_test_set, config)
    return train_test_set, spatial_test_set
    
def qualitative_results(model, test_set, out_dir, weekday=2, target_times=None):
    '''
    Shows qualitative results. The given test_set is filtered for the given weekday 
    and a number of time steps specified in target_times.
    The default args are the ones used in the paper.
    '''
    if target_times is None:
        target_times = [0*12, 6*12, 12*12, 18*12]
    
    cmap = 'viridis'
    
    fig, ax = plt.subplots(2, len(target_times), figsize=(18, 9))
    plt.tight_layout()
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    inputs = [ None for i in target_times ]

    for s in test_set:
        if s['weekday'][0] == 2 and  s['time_idx'][0] in target_times:
            idx = target_times.index(s['time_idx'][0])
            inputs[idx] = s

    for i, traffic_input in enumerate(inputs):
        city = traffic_input['city'][0].numpy().decode()
        day = weekdays[traffic_input['weekday'][0]]
        hour = traffic_input['time_idx'][0].numpy() // 12
        minute = traffic_input['time_idx'][0].numpy() % 12
        print('Data sample')
        print(' * City:    ', city)
        print(' * Time:    ', '%.2d:%.2d' % (hour, minute))
        print(' * Weekday: ', day)
        pred = model._call(traffic_input)

        pred_img = graph_to_image(pred, test_set.get_img_mask(traffic_input['city'][0])).numpy()
        target_img = graph_to_image(traffic_input['graph']['target_nodes'], 
                                    test_set.get_img_mask(traffic_input['city'][0])).numpy()

        pred_img[pred_img * 255 < 0.9] *= 0

        ax[0][i].imshow(np.log(pred_img.sum(-1)*255 + 1), cmap=cmap)
        ax[0][i].get_xaxis().set_ticks([])
        ax[0][i].get_yaxis().set_ticks([])
        ax[1][i].imshow(np.log(target_img.sum(-1)*255 + 1), cmap=cmap)
        ax[1][i].get_xaxis().set_ticks([])
        ax[1][i].get_yaxis().set_ticks([])
        ax[0][i].set_title('%s %.2d:%.2d' % (day, hour, minute))

    fig.add_axes()
    ax[0][0].set_ylabel('Prediction')
    ax[1][0].set_ylabel('Ground Truth')
    fig.savefig(os.path.join(out_dir, 'qualitative_results_%s.pdf' % weekdays[weekday]), dpi=300)
    plt.close()

def quantitative_evaluation(model, dataset):
    results = []
    progbar = tf.keras.utils.Progbar(dataset.size)
    
    for i, traffic_data in enumerate(dataset):
        metrics_dict = model.quantitative_evaluation_step(traffic_data)
        metrics_dict.update({
            'city' : traffic_data['city'][0],
            'date' : traffic_data['date'][0],
            'time_idx' : traffic_data['time_idx'][0],
            'weekday'  : traffic_data['weekday'][0],
        })
        results.append(metrics_dict)
        progbar.add(1)
        
    return results

def plot_metric_by(df, groupby, metric='score', ax=None, color='C0', label=None):
    label = '' if label is None else label
    plt.style.use('seaborn')
    mean_score = df.groupby(groupby).mean()[metric]
    std_score = df.groupby(groupby).std()[metric]
    median_score = df.groupby(groupby).median()[metric]

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.fill_between(mean_score.index, mean_score - std_score, mean_score + std_score, alpha=0.2, color=color)
    ax.plot(mean_score, color=color, label='mean %s' % label)
    ax.plot(median_score, color=color, label='median %s' % label, linestyle='--')

    ax.legend()
    ax.get_xticks()
    return ax

def compare_train_flipped(model, train_test_set, spatial_test_set, out_dir):
    '''
    Creates result plots that compare model performance on the training set vs. the
    (horizontally and vertically) flipped train set quntitatively to assess
    spatial generalization.
    '''
    quantithative_results_train = quantitative_evaluation(model, train_test_set)
    quantitative_results_spatial = quantitative_evaluation(model, spatial_test_set)
    
    train_eval = pd.DataFrame(data=quantithative_results_train).applymap(
        lambda x: x.numpy().decode() if isinstance(x.numpy(), bytes) else x.numpy())
    spatial_eval = pd.DataFrame(data=quantitative_results_spatial).applymap(
        lambda x: x.numpy().decode() if isinstance(x.numpy(), bytes) else x.numpy())
    
    plt.style.use('seaborn')
    mean_score = train_eval.groupby('weekday').aggregate(np.mean)['score']
    std_score = train_eval.groupby('weekday').aggregate(np.std)['score']

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 6))

    sort = [6, 4, 2, 1, 0, 5, 7, 3]

    groups = np.array(list(train_eval.groupby('city').groups.keys()))[sort]

    # plot overal mean marker
    ax[0].axhline(train_eval['score'].mean(), linestyle='--', color='C0')
    ax[0].axhline(spatial_eval['score'].mean(), linestyle='--', color='C1')

    bp = ax[0].boxplot(train_eval.groupby('city')['score'].apply(list).values[sort],
              positions=np.array(range(len(groups)))*2.0-0.4, patch_artist=True)
    bp1 = ax[0].boxplot(spatial_eval.groupby('city')['score'].apply(list).values[sort],
              positions=np.array(range(len(groups)))*2.0+0.4, patch_artist=True)

    ax[0].text(-0.5, train_eval['score'].mean()+3., train_eval['score'].mean().round(2), color='C0', 
            verticalalignment='bottom')
    ax[0].text(0.5, spatial_eval['score'].mean()+3., spatial_eval['score'].mean().round(2), color='C1', 
            verticalalignment='bottom')

    for prop in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(bp[prop], color='C0', linewidth=1.5)
        plt.setp(bp1[prop], color='C1', linewidth=1.5)
    plt.setp(bp['boxes'], facecolor=mpl.colors.to_rgba('C0')[:-1] + (0.4,))
    plt.setp(bp['fliers'], markeredgecolor='C0')
    plt.setp(bp1['boxes'], facecolor=mpl.colors.to_rgba('C1')[:-1] + (0.4,))
    plt.setp(bp1['fliers'], markeredgecolor='C1')

    ax[0].plot([], color='C0', label='train-set')
    ax[0].plot([], color='C1', label='train-set (flipped)')
    ax[0].legend(loc='upper left')

    ax[0].set_xticks(range(len(groups)*2)[::2])
    ax[0].set_xticklabels(groups, rotation=30)
    ax[0].set_title('MSE by City')
    ax[0].set_ylabel('MSE')

    plot_metric_by(train_eval, 'time_idx', color='C0', label='mse', ax=ax[1])
    plot_metric_by(spatial_eval, 'time_idx', color='C1', label='mse (flipped)', ax=ax[1])
    ax[1].set_title('MSE by Time Step')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('MSE')
    ax[1].set_xticks(train_eval.groupby('time_idx').mean()['score'].index[::2])
    ax[1].set_xticklabels([ datetime.time(hour=i // 12 + 1, minute=i % 12 * 5).strftime('%H:%M') 
                         for i in train_eval.groupby('time_idx').mean()['score'].index[::2] ], 
                       rotation=0)


    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_by_city_and_daytime.pdf'), dpi=300)
    plt.close()
    
    
if __name__ == '__main__':
    parser = create_cli_parser()

    parser.add_argument('--n_days', type=int, nargs='?', default=30,
                    help='The number of days to compute the metrics over'
                         ', defaults to 30.')
    parser.add_argument('--starting_date', type=str, nargs='?', default='2019-04-01',
                    help='The starting date from which the results are generated over --n_days'
                         ', defaults to 2019-04-01.')

    args = parser.parse_args()
    config = DEFAULTS
    config.update(vars(args))
    
    model, op = init_model(config['model'][0], config)
    print(f'[Results] Generating results for model {config["model"][0]}')
    print(f'[Results]    * Using checkpoint: {config["checkpoint"]}')
    print(f'[Results]    * Model step:       {model.global_step.numpy()}')
    
    output_dir = os.path.join('results', model.id)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    test_set = create_one_city_dataset(config)
    qualitative_results(model, test_set, out_dir=output_dir)
    
    train_test_set, spatial_test_set = create_quantitative_eval_dataset(config, 
        n_days=config['n_days'], starting_date=config['starting_date'])
    compare_train_flipped(model, train_test_set, spatial_test_set, out_dir=output_dir)