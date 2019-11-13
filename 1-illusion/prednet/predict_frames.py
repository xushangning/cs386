import os
import numpy as np
from six.moves import cPickle
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
# from data_utils import SequenceGenerator

from preprocess import *

matplotlib.use('Agg')

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
# downloaded from: https://www.dropbox.com/s/iutxm0anhxqca0z/model_data_keras2.zip?dl=0
WEIGHTS_DIR = './model_data_keras2/'

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = './results/'


def init_model(nt=10):
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    # test_file = os.path.join(DATA_DIR, 'X_test.hkl')
    # test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_file)

    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)
    return test_model


# predict next frame
# fixed means predict based on fixed input
# recur means the predicted result is used as next frame of input
def predict(test_model, X_test, nt, mode='fixed', initial=2, batch_size=10):
    if mode == 'fixed':
        X_hat = test_model.predict(X_test.transpose((0, 1, 4, 2, 3)), batch_size)
        return X_hat.transpose((0, 1, 3, 4, 2))
    if mode == 'feedback':
        new_test = X_test.transpose((0, 1, 4, 2, 3)).copy()
        for i in range(initial, nt):
            print('Predicting the {}-th frame'.format(i + 1))
            X_hat = test_model.predict(new_test, batch_size)
            new_test[:, i, :, :, :] = X_hat[:, -1, :, :, :]
        return X_hat.transpose((0, 1, 3, 4, 2))


# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
def mse_error(X_test, X_hat):
    mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)
    if not os.path.exists(RESULTS_SAVE_DIR):
        os.mkdir(RESULTS_SAVE_DIR)
    f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Previous Frame MSE: %f" % mse_prev)
    f.close()


# Plot some predictions
def plot_pred(X_test, X_hat, nt=10, n_plot=100):
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize=(nt, 2 * aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
    if not os.path.exists(plot_save_dir):
        os.mkdir(plot_save_dir)
    plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
    for i in plot_idx:
        for t in range(nt):
            plt.subplot(gs[t])
            plt.imshow(X_test[i, t], interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                            labelbottom='off',
                            labelleft='off')
            if t == 0:
                plt.ylabel('Actual', fontsize=10)

            plt.subplot(gs[t + nt])
            plt.imshow(X_hat[i, t], interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                            labelbottom='off',
                            labelleft='off')
            if t == 0:
                plt.ylabel('Predicted', fontsize=10)

        plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
        plt.clf()


def main():
    # number of frames in the sequence
    nt = 5
    # batch_size = 8
    test_model = init_model(nt)
    X_test = get_static_test_data('test_data/static', nt)
    X_hat = predict(test_model, X_test, nt, 'feedback', 2)
    # print(X_hat.shape)
    # print(X_test.shape)
    mse_error(X_test, X_hat)
    plot_pred(X_test, X_hat, nt)


if __name__ == '__main__':
    main()
