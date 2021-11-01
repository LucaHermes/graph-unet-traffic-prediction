from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import scipy.signal
import data


# --------- T4C Evaluation methods ---------------

def create_city_mask(static_data):
    '''
    Replicated the behavior of the mask generation in the
    official traffic4cast repo:
    https://github.com/iarai/NeurIPS2021-traffic4cast/blob/master/competition/scorecomp/scorecomp.py
    static_data shape: [batch, 9, height, width]
    '''
    mask = np.where(static_data[:,0] > 0, 1, 0)
    # create a broadcastable mask
    # mask shape: [height, width, 1]
    mask = mask[..., np.newaxis]
    mask = mask[:, np.newaxis]
    return mask

def mse_fn(ground_truth, prediction, axis=None):
    '''
    Standard mse term.
    '''
    return np.mean((ground_truth - prediction)**2, axis=axis)

def compute_mse(ground_truth, prediction, mask=None):
    '''
    Replicated the behavior of the compute_mse method in the
    official traffic4cast repo:
    https://github.com/iarai/NeurIPS2021-traffic4cast/blob/master/competition/scorecomp/scorecomp.py
    The difference is that this implementation is in full numpy.

    Parameters:
    ----------
        ground_truth : np.array
            Ground truth in image format: [n_samples, time, height, width, channels].
            n_samples corresponds to the number of samples to compute the score for.
            Can also be the batch size used in training/evaluation.
        prediction : np.array
            Prediction in image format: [n_samples, time, height, width, channels].
            n_samples corresponds to the number of samples to compute the score for.
            Can also be the batch size used in training/evaluation.
        mask : np.array
            Street mask (1st channel of the static data) in image format, 
            expected shape: [batch, height, width].
    Returns:
    ----------
        Dictionary with metric results.
    '''
    scores = {}
    ground_truth = np.array(ground_truth, dtype=np.float32)
    ground_truth_vol = ground_truth[..., data.VOL_CHANNELS]
    ground_truth_speed = ground_truth[..., data.SPEED_CHANNELS]
    prediction = np.array(prediction, dtype=np.float32)
    prediction_vol = prediction[..., data.VOL_CHANNELS]
    prediction_speed = prediction[..., data.SPEED_CHANNELS]

    scores['mse'] = mse_fn(ground_truth, prediction)
    scores['mse_volumes'] = mse_fn(ground_truth_vol, prediction_vol)
    scores['mse_speeds'] = mse_fn(ground_truth_speed, prediction_speed)

    if mask is not None:
        mask = np.array(mask, dtype=np.float32)
        mse_masked_base = mse_fn(ground_truth * mask, prediction * mask, axis=(-1,-2,-3,-4))
        mse_masked_vol_base = mse_fn(ground_truth_vol * mask, prediction_vol * mask, axis=(-1,-2,-3,-4))
        mse_masked_speed_base = mse_fn(ground_truth_speed * mask, prediction_speed * mask, axis=(-1,-2,-3,-4))
        scores['mse_masked_base'] = mse_masked_base.mean()
        mask_ratio = np.count_nonzero(mask, axis=(-1,-2,-3)) / np.prod(mask.shape[1:])
        scores['mask_ratio'] = mask_ratio
        scores['mse_masked'] = np.mean(mse_masked_base / mask_ratio)
        scores['mse_masked_volumes_base'] = mse_masked_vol_base.mean()
        scores['mse_masked_volumes'] = np.mean(mse_masked_vol_base / mask_ratio)
        scores['mse_masked_speeds_base'] = mse_masked_speed_base.mean()
        scores['mse_masked_speeds'] = np.mean(mse_masked_speed_base / mask_ratio)

    return scores



def compute_score(ground_truth, prediction, static_data=None, mask=None, batch_size=32):
    '''
    Replicated the behavior of the score computation in the
    official traffic4cast repo, what they are doing for all cities at once,
    this method does for general ground truth and prediction arrays:
    https://github.com/iarai/NeurIPS2021-traffic4cast/blob/master/competition/scorecomp/scorecomp.py

    Computes the score considering the given ground truth and model predictions.
    static_data is used to create a mask so only values are considered where 
    there is actually a street.

    Parameters:
    ----------
        ground_truth : np.array
            Ground truth in image format: [n_samples, time, height, width, channels].
            n_samples corresponds to the number of samples to compute the score for.
            Can also be the batch size used in training/evaluation.
        prediction : np.array
            Prediction in image format: [n_samples, time, height, width, channels].
            n_samples corresponds to the number of samples to compute the score for.
            Can also be the batch size used in training/evaluation.
        static_data : np.array
            Static data for the cities in the batch used for masking, 
            expected shape: [batch, 9, height, width].
            (either static_data or mask must be given)
        mask : np.array
            Street mask (1st channel of the static data) in image format, 
            expected shape: [batch, height, width].
            (either static_data or mask must be given)
        batch_size : int
            If n_samples is large and the score computation would exceed the memory 
            limitations, batch_size can be used to limit the samples processed concurrently.
    Returns:
    ----------
        Dictionary with metric results.

    '''
    score = 0
    scores_dict = defaultdict(float)
    n_batches = len(ground_truth) // batch_size

    for b in range(max(1, n_batches)):
        batch_start = b * batch_size
        batch_end = (b + 1) * batch_size
        if static_data is not None:
            img_mask = create_city_mask(static_data[batch_start:batch_end])
        else:
            if mask is None:
                raise ValueError('Either static_data or mask must be given as a parameter.')
            img_mask = mask[batch_start:batch_end, np.newaxis]
        batch_scores = compute_mse(
            ground_truth[batch_start:batch_end],
            prediction[batch_start:batch_end],
            mask=img_mask)
        batch_score = batch_scores['mse']
        score += batch_score

        for k, v in batch_scores.items():
            scores_dict[k] += v

    for k in scores_dict.keys():
        scores_dict[k] /= max(1, n_batches)

    score /= max(1, n_batches)
    # convert defaultdict back to dict
    scores_dict = dict(scores_dict)

    return score, scores_dict





# --------- Custom Evaluation methods ---------------

def masked_mse(y_true, y_pred, mask):
    '''
    Computes the masked mse for the image data.
    '''
    mask = tf.cast(mask, tf.float32)
    mask_dims = tf.cast(tf.reduce_prod(tf.shape(mask)[1:]), tf.float32)
    mask_ratio = tf.reduce_sum(1. - mask, axis=(-1, -2)) / mask_dims
    mask = mask[...,tf.newaxis]
    mse_base = tf.reduce_mean(tf.keras.losses.MSE(y_true * mask, y_pred * mask))
    return mse_base / mask_ratio

def compute_metrics(y_true, y_pred, data_scale=255, mask=None, score_only=False):
    '''
    Computes scaled and unscaled training metrics.

    Parameters:
    ----------
        y_true : Tensor
            Ground truth in image format or in graph format. 
            image format: [batch, height, width, channels*time] or
            graph format: [nodes, features*time] 
        y_pred : Tensor
            Prediction in image format or in graph format.
            image format: [batch, height, width, channels*time] or
            graph format: [nodes, features*time] 
        data_scale : int, float
            Scalar that will be applied to the data for the scaled plots, defaults to 255.
    Returns:
    ----------
        Dictionary with metric results.
    '''
    y_true_scaled = y_true * data_scale
    y_pred_scaled = y_pred * data_scale

    if mask is not None:
        # if samples are images, use masked mse
        mse_method = masked_mse
    else:
        # if samples are graphs, use standard mse
        mse_method = lambda t, p, m: tf.keras.losses.MSE(t, p)

    mse = tf.reduce_mean(mse_method(y_true, y_pred, mask))
    mse_scaled = tf.reduce_mean(mse_method(y_true_scaled, y_pred_scaled, mask))
    
    if score_only:
        return { 'mse_scaled' : mse_scaled }
    
    mae = tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred))
    mae_scaled = tf.reduce_mean(
        tf.keras.losses.MAE(y_true_scaled, y_pred_scaled))
    mae_scaled_speed = tf.reduce_mean(
        tf.keras.losses.MAE(
            y_true_scaled[...,data.SPEED_CHANNEL_SLICE], 
            y_pred_scaled[...,data.SPEED_CHANNEL_SLICE]))
    mae_scaled_vol = tf.reduce_mean(
        tf.keras.losses.MAE(
            y_true_scaled[...,data.VOL_CHANNEL_SLICE], 
            y_pred_scaled[...,data.VOL_CHANNEL_SLICE]))
    rmse_scaled = tf.sqrt(mse_scaled)

    metrics = {
        'mse' : mse,
        'mae' : mae,
        'mse_scaled' : mse_scaled,
        'mae_scaled' : mae_scaled,
        'mae_scaled_speed'  : mae_scaled_speed,
        'mae_scaled_volume' : mae_scaled_vol,
        'rmse_scaled' : rmse_scaled,
    }

    y_true_scaled = tf.split(y_true_scaled, 12, axis=-1)
    y_pred_scaled = tf.split(y_pred_scaled, 12, axis=-1)

    for i in [0, 1, 2, 5, 8, 11]:
        mins = (i+1) * 5
        key = f'mse_scaled_{mins}_min'
        e = tf.reduce_mean(mse_method(
            y_true_scaled[i],
            y_pred_scaled[i], 
            mask))
        metrics[key] = e

    return metrics


positive_log_formatter= FuncFormatter(lambda x, y: '$10^{%.1f}$' % 
    np.log10(np.e**x) 
    if np.log10(np.e**x) < -0.1 
    else round(2**x, 3))


def plot_log_img(img, fig=None, ax=None, plot_colorbar=True, eps=1e-5, cmap='jet', vmin=0, vmax=1):
    '''
    Generates a logarithmic image with a colorbar.
    Negative values in img will be clipped to eps so
    log result is valid.
    '''
    if ax is None:
        fig, ax = plt.subplots()

    img_pos = np.maximum(0, img)
    log_vmin = np.log(np.maximum(eps, vmin))
    log_vmax = np.log(vmax)

    im = ax.imshow(np.log(img_pos + eps), cmap=cmap, vmin=log_vmin, vmax=log_vmax)#, norm=matplotlib.colors.LogNorm(vmax=speed_err.max()))
    
    ax.axis('off')
    #delta_log = np.log10(2**cb.vmax) - np.log10(2**cb.vmin)
    #n_ticks = int(round(delta_log))
    cb = fig.colorbar(im)
    cb.set_ticks(np.linspace(cb.vmin, cb.vmax, 7))
    cb.ax.yaxis.set_major_formatter(positive_log_formatter)
    return fig

def create_eval_plots(y_true, y_pred, x=None, mask=None, n=1, data_scale=255, seed=None, notes=None):
    '''
    Creates plots of the true and predicted 
    trajectory for n single cells.

    Parameters:
    ----------
        y_true : Tensor
            Ground truth in image format: [batch, time, height, width, channels] 
        y_pred : Tensor
            Prediction in image format: [batch, time, height, width, channels]
        x: Tensor
            model input in image format: [batch, time, height, width, channels]
            (optional)
        mask : Tensor
            Street mask (1st channel of the static data) in image format: [batch, height, width].
            This method uses the mask only to select interesting areas in the data, if no mask is
            given, it is likely to sample from a region without any roads.
            (optional)
        n : int
            Number of plots, defaults to 1.
        data_scale : int, float
            Scalar that will be applied to the data for the scaled plots, defaults to 255.
        seed : int
            Seed may be passed to always sample the same indices of the data
            to generate the plots from, defaults to None.
        notes : str
            Notes will be added to the plots as text (optional).

    Returns
    ----------
        Dictionary of plots.
    '''
    # ----- Config ------
    patch_size = 100
    patch_offset_x = 150
    patch_offset_y = 150
    fft_window = 4
    # -------------------

    shape = y_true.shape
    b, target_steps, h, w = shape[0], shape[1], shape[-3], shape[-2]
    n_cells = b * h * w

    patch_x = slice(patch_offset_x, patch_offset_x+patch_size, 1)
    patch_y = slice(patch_offset_y, patch_offset_y+patch_size, 1)

    if mask is None:
        select_cells = np.random.RandomState(seed).randint(0, n_cells, n)
        b_idx = select_cells // (w * h)
        h_idx = select_cells // w // b
        w_idx = select_cells % w
    else:
        idx = np.transpose(np.where((mask*y_true.sum((-1,-4))) != 0))
        selector = np.random.RandomState(seed)
        try:
            b_idx, h_idx, w_idx = idx[selector.choice(max(len(idx), 1), n)].T
        except:
            print('WARNING: Target image is completely empty.')
            b_idx, h_idx, w_idx = [np.array([0])]*3

    true_traj = y_true * data_scale
    pred_traj = y_pred * data_scale

    if x is not None:
        x = x * data_scale
        true_traj = np.concatenate((x, true_traj), axis=1)
        pred_xs = np.arange(x.shape[1], x.shape[1]+target_steps)


    # clear open plots
    plt.close('all')
    plots = {}
    
    target_gt = true_traj[:,-target_steps:]
    error = np.mean(np.abs(target_gt - pred_traj), axis=1)
    true_spatial_sum = true_traj.sum((2, 3))
    pred_spatial_sum = pred_traj.sum((2, 3))
    true_temporal_sum = true_traj.sum(1)
    pred_temporal_sum = pred_traj.sum(1)

    # create windows of length fft_window along the temporal axis for spectral plot
    # 1. cut out patches and sum channels
    true_patch_speed_sum = target_gt[:,:,patch_y, patch_x, data.SPEED_CHANNEL_SLICE].sum(-1)
    pred_patch_speed_sum = pred_traj[:,:,patch_y, patch_x, data.SPEED_CHANNEL_SLICE].sum(-1)
    true_patch_speed_sum = true_patch_speed_sum.astype(np.float32)
    pred_patch_speed_sum = pred_patch_speed_sum.astype(np.float32)

    # generate temporal windows of length fft_window
    true_patch_speed_sum = np.stack([ true_patch_speed_sum[:,t:t+fft_window] 
        for t in range(target_steps-fft_window) ], axis=1)
    pred_patch_speed_sum = np.stack([ pred_patch_speed_sum[:,t:t+fft_window] 
        for t in range(target_steps-fft_window) ], axis=1)

    # apply pre-windowing (empirically gives a clearer view)
    hann_window = scipy.signal.windows.hann(fft_window).reshape([1,1,fft_window,1,1])
    true_patch_speed_sum = (true_patch_speed_sum != 0) * hann_window
    pred_patch_speed_sum = (pred_patch_speed_sum != 0) * hann_window
    true_patch_speed_spectrum = scipy.signal.welch(true_patch_speed_sum, axis=2, nperseg=fft_window)[1]
    pred_patch_speed_spectrum = scipy.signal.welch(pred_patch_speed_sum, axis=2, nperseg=fft_window)[1]
    
    for n, b in enumerate(np.unique(b_idx)):
        batch_mask = b_idx == b

        # wandb needs separate matplotlib figures
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()
        fig10, ax10 = plt.subplots()
        fig11, ax11 = plt.subplots(2, 1, figsize=(14, 4))
        fig12, ax12 = plt.subplots()
        fig13, ax13 = plt.subplots()
        fig14, ax14 = plt.subplots()
        fig15, ax15 = plt.subplots()

        # plot sum over whole city as line graph
        ax1.plot(true_spatial_sum[b,...,data.SPEED_CHANNELS].sum(0), linestyle='--', label='true')
        ax2.plot(true_spatial_sum[b,...,data.VOL_CHANNELS].sum(0), linestyle='--', label='true')
        ax1.plot(pred_xs, pred_spatial_sum[b,...,data.SPEED_CHANNELS].sum(0), label='pred')
        ax2.plot(pred_xs, pred_spatial_sum[b,...,data.VOL_CHANNELS].sum(0), label='pred')
        ax1.legend()
        ax2.legend()

        # show image of the error over the city
        speed_err = error[b,...,data.SPEED_CHANNELS].mean(0)
        volum_err = error[b,...,data.VOL_CHANNELS].mean(0)
        im1 = ax3.imshow(speed_err, cmap='jet')
        im2 = ax4.imshow(volum_err, cmap='jet')

        fig3.colorbar(im1)
        fig4.colorbar(im2)
        
        ax3.axis('off')
        ax4.axis('off')

        fig5 = plot_log_img(speed_err, fig=fig5, ax=ax5, vmin=speed_err.min(), vmax=speed_err.max())
        fig6 = plot_log_img(volum_err, fig=fig6, ax=ax6, vmin=volum_err.min(), vmax=volum_err.max())

        true_speed_img = true_temporal_sum[b,...,data.SPEED_CHANNELS].sum(0)
        true_volum_img = true_temporal_sum[b,...,data.VOL_CHANNELS].sum(0)
        pred_speed_img = pred_temporal_sum[b,...,data.SPEED_CHANNELS].sum(0)
        pred_volum_img = pred_temporal_sum[b,...,data.VOL_CHANNELS].sum(0)

        speed_vmin = min(true_speed_img.min(), pred_speed_img.min())
        volum_vmin = min(true_volum_img.min(), pred_volum_img.min())
        speed_vmax = max(true_speed_img.max(), pred_speed_img.max())
        volum_vmax = max(true_volum_img.max(), pred_volum_img.max())

        # values <= 0 will be clipped to eps
        fig7  = plot_log_img(true_speed_img, fig=fig7, ax=ax7, eps=1e-8, vmin=speed_vmin, vmax=speed_vmax)
        fig8  = plot_log_img(true_volum_img, fig=fig8, ax=ax8, eps=1e-8, vmin=volum_vmin, vmax=volum_vmax)
        fig9  = plot_log_img(pred_speed_img, fig=fig9, ax=ax9, eps=1e-8, vmin=speed_vmin, vmax=speed_vmax)
        fig10 = plot_log_img(pred_volum_img, fig=fig10, ax=ax10, eps=1e-8, vmin=volum_vmin, vmax=volum_vmax)

        # mark the single cells that will be plotted in the subsequent for loop
        for h, w in zip(h_idx[batch_mask], w_idx[batch_mask]):
            ax3.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax4.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax5.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax6.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax7.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax8.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax9.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)
            ax10.scatter(w, h, s=60, facecolors='none', edgecolors='w', linewidths=2)

        # concat spectral images into a single wide image
        true_spectrum = np.concatenate(true_patch_speed_spectrum[0], axis=-1).sum(0)
        pred_spectrum = np.concatenate(pred_patch_speed_spectrum[0], axis=-1).sum(0)
        marker_positions = np.arange(patch_size, patch_size*(target_steps-fft_window), patch_size)

        # plot pixelwise power spectrum
        i1 = ax11[0].imshow(true_spectrum, cmap=plt.cm.gnuplot)
        i2 = ax11[1].imshow(pred_spectrum, cmap=plt.cm.gnuplot)
        ax11[0].vlines(marker_positions, 0, patch_size-1, color='darkgrey')
        ax11[1].vlines(marker_positions, 0, patch_size-1, color='darkgrey')
        ax11[0].set_ylabel('True Spectrum')
        ax11[1].set_ylabel('Predicted Spectrum')
        ax11[0].set_yticks([])
        ax11[0].set_xticks([])
        ax11[1].set_yticks([])
        ax11[1].set_xticks([])

        im12 = ax12.imshow(true_speed_img, cmap='gnuplot')
        im13 = ax13.imshow(true_volum_img, cmap='gnuplot')
        im14 = ax14.imshow(pred_speed_img, cmap='gnuplot')
        im15 = ax15.imshow(pred_volum_img, cmap='gnuplot')
        ax12.axis('off')
        ax13.axis('off')
        ax14.axis('off')
        ax15.axis('off')
        fig12.colorbar(im12)
        fig13.colorbar(im13)
        fig14.colorbar(im14)
        fig15.colorbar(im15)

        #cax1 = fig11.add_axes((1., 0.53, 0.008, 0.435))
        #cax2 = fig11.add_axes((1., 0.04, 0.008, 0.440))

        #fig11.colorbar(i1, cax=cax1)
        #fig11.colorbar(i2, cax=cax2)
        fig11.tight_layout()

        plots.update({
            f'speed_b{n}'           : fig1, 
            f'volume_b{n}'          : fig2,
            f'speed_mae_b{n}'       : fig3, 
            f'volume_mae_b{n}'      : fig4,
            f'speed_mae_log_b{n}'   : fig5, 
            f'volume_mae_log_b{n}'  : fig6, 
            f'speed_true_log_b{n}'  : fig7, 
            f'volume_true_log_b{n}' : fig8, 
            f'speed_pred_log_b{n}'  : fig9, 
            f'volume_pred_log_b{n}' : fig10, 
            f'speed_fft_b{n}'       : fig11, 
            f'speed_true_b{n}'      : fig12, 
            f'volume_true_b{n}'     : fig13, 
            f'speed_pred_b{n}'      : fig14, 
            f'volume_pred_b{n}'     : fig15,
        })

    # plot errors of individual pixels
    true_nodes = true_traj[b_idx, :, h_idx, w_idx, :]
    pred_nodes = pred_traj[b_idx, :, h_idx, w_idx, :]

    #                     blue        orange     green      red
    direction_colors = ['#0099ff', '#ff9900', '#33cc33', '#ff6666']

    for i, (gt, pred) in enumerate(zip(true_nodes, pred_nodes)):
        # create speed plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        plt.tight_layout()

        if notes is not None:
            for a in [ax1, ax2]:
                a.text(.05, .05, notes, 
                    horizontalalignment='center', 
                    verticalalignment='bottom',
                    transform=a.transAxes)

        for s, v, c in zip(data.SPEED_CHANNELS, data.VOL_CHANNELS, direction_colors):
            ax1.plot(gt[...,s], color=c, linestyle='--', 
                label='true ' + data.CHANNEL_LABELS[s] if i == 0 else None)
            ax1.plot(pred_xs, pred[...,s], color=c, 
                label='pred ' + data.CHANNEL_LABELS[s] if i == 0 else None)
            ax2.plot(gt[...,v], color=c, linestyle='--', 
                label='true ' + data.CHANNEL_LABELS[v] if i == 0 else None)
            ax2.plot(pred_xs, pred[...,v], color=c, 
                label='pred ' + data.CHANNEL_LABELS[v] if i == 0 else None)
        ax1.legend()
        ax2.legend()

        plots.update({
            f'speed_prediction_{i}'  : fig1, 
            f'volume_prediction_{i}' : fig2
        })

    return plots





if __name__ == '__main__':
    import collections
    import h5py

    # test with results received from the original implementation
    file = 'data/raw/custom/2019-03-25_BARCELONA_8ch.h5'
    static_file = 'data/raw/custom/BARCELONA_static.h5'

    true_scores = {
        'mask_ratio': 0.443,
        'mse': 46.21175003051758,
        'mse_masked': 104.11118277039539,
        'mse_masked_base': 46.121253967285156,
        'mse_masked_speeds': 207.8712583933673,
        'mse_masked_speeds_base': 92.08696746826172,
        'mse_masked_volumes': 0.35111454754866006,
        'mse_masked_volumes_base': 0.1555437445640564,
        'mse_speeds': 92.26731872558594,
        'mse_volumes': 0.15617172420024872
    }

    test_data = []
    test_offsets = [0, 1, 2, 3, 6, 9, 12]
    test_ids = np.array([
        190, 260, 163, 271,  46,  34,  35, 241, 168, 247, 103, 231, 116,
        43, 174,  97, 198,  55, 191,  14,   4, 155, 181,  80,  79,  34,
        222, 233,  28,  30, 212,  87,  48,  31, 192, 246, 113,  86, 219,
        249, 126, 157, 264, 129, 207, 186, 228, 123, 106,  13, 151,  57,
        229, 219,  46,  14, 102, 268, 222,  45,  16,  42,  56,  53
    ])

    print('Testing the evaluation protocol agains values received from',
          'the original implementation.')

    with h5py.File(file, 'r') as f:
        for tid in test_ids:
            test_data.append(f['array'][tid + test_offsets])
    with h5py.File(static_file, 'r') as f:
        static_data = np.array(f['array'])
            
    test_data = np.stack(test_data)

    # do zero_velocity prediction
    gt = test_data[:,1:]
    pred = test_data[:,:1].repeat(6, axis=1)
    static_data = static_data[np.newaxis].repeat(len(test_ids), axis=0)

    score, scores_dict = compute_score(gt, pred, static_data, batch_size=4)

    print('Scores of the naive zero-velocity baseline:')
    for k, v in scores_dict.items():
        v = v if not isinstance(v, collections.Iterable) else np.mean(v)
        print('\t%-23s : %.8f' % (k, v))
    print('\t%-23s : %.8f' % ('SCORE', scores_dict['mse_masked']))

    assert np.all([ np.isclose(np.mean(v), np.mean(scores_dict[k])) for k, v in true_scores.items() ])
    
    print('The values are matching.')
