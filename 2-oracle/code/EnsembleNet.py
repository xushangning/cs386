import numpy as np
import cv2
import tensorflow as tf
import keras.layers as L
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import *
from keras.models import load_model

from Dataset import Dataset
from utils import VisualizationCallback
from OracleNet import cat10_model_conv
from OracleNet import cat40_model_conv
from template_matching import TemplateMatch
from utils import plot_confusion_matrix


class EnsembleNet:
    def __init__(self, cat=40, filepath=""):
        assert cat in [10, 40], "Number of categories can only be 10 or 40"

        self.cat = cat

        # if cat == 10:
        #     # self.conv_net = cat10_model_conv()
        #     self.conv_net = load_model('./model/weights_norm_cat10.hdf5')
        # else:
            # self.conv_net = cat40_model_conv()
        self.conv_net = load_model('./model/weights_norm_cat40.hdf5')

        self.tm = TemplateMatch(method='CCORR_NORMED')
        self.tm.load_model('./model/templ.pkl')
        # self.tm2 = TemplateMatch(method='MSE_NORMED')
        # self.tm2.load_model('./model/templ.pkl')

        if filepath:
            self.ens = load_model(filepath)
        else:
            self.ens = self.create_ensemble()

    def create_ensemble(self):
        print('creating ensemble')
        model = Sequential()
        model.add(L.Dense(self.cat, activation='softmax', input_shape=(80,)))

        # set up optimizer and compile model
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print('ensemble compiled')
        return model

    def process_input(self, X):
        print('process input...')
        a = self.conv_net.predict(X / 255)
        b = self.tm(X)
        ens_input = np.concatenate((a, b), axis=1)
        print('ens_input', ens_input.shape)
        return ens_input

    def __call__(self, X):
        return self.ens.predict(self.process_input(X))


class EnsembleNetTrainer(object):
    def __init__(self, test_size=0.1, autosave_path='./tmp/weights.{val_loss:.2f}.hdf5', vis_on_batch=False):
        """

        :param test_size: ratio of test (validation) set
        :param autosave_path: path to auto save training results
        :param vis_on_batch: whether to record visualization on batch
        """
        # load original data
        X_train, y_train = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['test'])
        X_val, y_val = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['test'], inclusive=True,
                                         normalize=True)
        X_val *= 255
        # X_train = X_train.reshape(X_train.shape + (1,))
        # X_val = X_val.reshape(X_val.shape + (1,))

        self.X_train = X_train
        self.X_val = X_val

        # covert to different categories
        # y40 = Dataset.to_onehot(y, 40)
        # y = Dataset.cat40_to_cat10(y)
        # y10 = Dataset.to_onehot(y, 10)
        # train test split
        # idx = list(range(y40.shape[0]))
        self.y_train = {}
        self.y_val = {}
        self.y_train[40] = Dataset.to_onehot(y_train, 40)
        y_train = Dataset.cat40_to_cat10(y_train)
        self.y_train[10] = Dataset.to_onehot(y_train, 10)
        self.y_val[40] = Dataset.to_onehot(y_val, 40)
        y_val = Dataset.cat40_to_cat10(y_val)
        self.y_val[10] = Dataset.to_onehot(y_val, 10)

        # initialize callbacks
        self.checkpointer = ModelCheckpoint(filepath=autosave_path, verbose=1, save_best_only=True)
        self.visualizer = VisualizationCallback(vis_on_batch)

    def train(self, ens_net, batch_size=32, epochs=100):
        num_cat = ens_net.cat

        ens_net.ens.fit(ens_net.process_input(self.X_train), self.y_train[num_cat], batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_data=(ens_net.process_input(self.X_val), self.y_val[num_cat]),
                        callbacks=[self.visualizer, self.checkpointer],
                        # verbose set to 0 only when running in console
                        verbose=0)

    def evaluate(self, ens_net, num_cat=40, confusion=True):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        print(ens_net.ens.evaluate(ens_net.process_input(self.X_val), self.y_val[num_cat]))
        if confusion:
            y_pred = ens_net(self.X_val).argmax(axis=1)
            plot_confusion_matrix(self.y_val[num_cat].argmax(axis=1), y_pred, num_cat)
            plt.show()
            print(y_pred)


if __name__ == '__main__':
    trainer = EnsembleNetTrainer()

    ens_model = EnsembleNet(40)
    trainer.train(ens_model, epochs=1000)

    # ens_model = EnsembleNet(40, './model/weights_ens_cat40.hdf5')
    # trainer.evaluate(ens_model)
