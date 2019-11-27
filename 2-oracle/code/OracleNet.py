#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras.layers as L
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import *
from keras.models import load_model

from sklearn.model_selection import train_test_split

from Dataset import Dataset
from utils import *


CLASS_NAMES = ["聨", "蚑", ""]

# training framework for oracle net
class OracleNetTrainer(object):
    def __init__(self, test_size=0.1, autosave_path='./tmp/weights.{val_loss:.2f}.hdf5', vis_on_batch=False):
        """

        :param test_size: ratio of test (validation) set
        :param autosave_path: path to auto save training results
        :param vis_on_batch: whether to record visualization on batch
        """
        # load original data
        X_train, y_train = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['val'])
        X_val, y_val = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['val'], inclusive=True)
        # X, y = Dataset.load_data(one_hot=False, num_cat=40)
        # normalize image to 0.0 - 1.0
        self.X_train = X_train / 255
        self.X_val = X_val / 255
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
        # self.X_train, self.X_val, y_train_idx, y_val_idx = train_test_split(X, idx, test_size=test_size, shuffle=True)
        # self.y_train[40], self.y_val[40] = y40[y_train_idx], y40[y_val_idx]
        # self.y_train[10], self.y_val[10] = y10[y_train_idx], y10[y_val_idx]
        # initialize callbacks
        self.checkpointer = ModelCheckpoint(filepath=autosave_path, verbose=1, save_best_only=True)
        self.visualizer = VisualizationCallback(vis_on_batch)

    def train(self, model, num_cat=10, batch_size=32, epochs=100):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        model.fit(self.X_train, self.y_train[num_cat], batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(self.X_val, self.y_val[num_cat]),
                  callbacks=[self.visualizer, self.checkpointer],
                  # verbose set to 0 only when running in console
                  verbose=0)

    def evaluate(self, model, num_cat=10):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        y_pred = model.predict(self.X_val).argmax(axis=1)
        plot_confusion_matrix(self.y_val[num_cat].argmax(axis=1), y_pred, range(num_cat))
        print(y_pred)


def cat10_model_simple():
    # build model structure
    model = Sequential()
    model.add(L.Flatten(input_shape=(64, 64)))
    model.add(L.Dense(64, activation='relu'))
    model.add(L.Dense(64, activation='relu'))
    model.add(L.Dense(10, activation='softmax'))
    # set up optimizer and compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def cat40_model_simple():
    # build model structure
    model = Sequential()
    model.add(L.Flatten(input_shape=(64, 64)))
    model.add(L.Dense(128, activation='relu'))
    model.add(L.Dense(64, activation='relu'))
    model.add(L.Dense(40, activation='softmax'))
    # set up optimizer and compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    trainer = OracleNetTrainer()

    model = load_model('./model/weights_simple_cat40.hdf5')
    trainer.evaluate(model, num_cat=40)
    plt.show()

    # model = load_model('./model/weights_simple_cat10.hdf5')
    # trainer.evaluate(model, num_cat=10)
    # plt.show()

    # model = cat10_model_simple()
    # trainer.train(model, num_cat=10, epochs=150)

    # model = cat40_model_simple()
    # trainer.train(model, num_cat=40, epochs=150)
