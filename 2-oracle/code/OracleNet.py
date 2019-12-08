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
from keras.models import Model

from sklearn.model_selection import train_test_split

from Dataset import Dataset
from utils import *

CLASS_NAMES = ["聨", "蚑", ""]


# training framework for oracle conv_net
class OracleNetTrainer(object):
    def __init__(self, test_size=0.1, autosave_path='./tmp/weights.{val_loss:.2f}.hdf5', vis_on_batch=False):
        """

        :param test_size: ratio of test (validation) set
        :param autosave_path: path to auto save training results
        :param vis_on_batch: whether to record visualization on batch
        """
        # load original data
        self.X_train, y_train = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['val', 'test'], normalize=True)
        self.X_val, y_val = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['val'], inclusive=True, normalize=True)
        # X_train = X_train.reshape(X_train.shape)
        # X_val = X_val.reshape(X_val.shape)
        # X, y = Dataset.load_data(one_hot=False, num_cat=40)
        # normalize image to 0.0 - 1.0
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

    def evaluate(self, model, num_cat=10, confusion=True):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        print(model.evaluate(self.X_val, self.y_val[num_cat]))
        if confusion:
            y_pred = model.predict(self.X_val).argmax(axis=1)
            plot_confusion_matrix(self.y_val[num_cat].argmax(axis=1), y_pred, num_cat)
            print(y_pred)

    @staticmethod
    def run_test(model, num_cat):
        X_test, y_test = Dataset.load_data(num_cat=num_cat, one_hot=True, filter_keys=['test'], inclusive=True,
                                           normalize=True)
        X_test[X_test < 0.5] = 0
        print(model.evaluate(X_test, y_test))
        y_pred = model.predict(X_test).argmax(axis=1)
        plot_confusion_matrix(y_test.argmax(axis=1), y_pred, num_cat)

    def error_matrix(self, model_cat10, model_cat40):
        y_pred_train = [model_cat10(self.X_train).argmax(axis=1),
                        model_cat40(self.X_train).argmax(axis=1)]
        y_pred_val = [model_cat10(self.X_val).argmax(axis=1),
                      model_cat40(self.X_val).argmax(axis=1)]
        mask_train = [(y_pred_train[0] == self.y_train[10]),
                      (y_pred_train[1] == self.y_train[40])]
        print(mask_train[0])
        mask_val = [(y_pred_val[0] == self.y_val[10]),
                    (y_pred_val[1] == self.y_val[40])]
        matrix_train = [sum(mask_train[0] & mask_train[1]), sum(mask_train[0] & (~mask_train[1])),
                        sum((~mask_train[0]) & mask_train[1]), sum((~mask_train[0]) & (~mask_train[1]))]
        matrix_val = [sum(mask_val[0] & mask_val[1]), sum(mask_val[0] & (~mask_val[1])),
                      sum((~mask_val[0]) & mask_val[1]), sum((~mask_val[0]) & (~mask_val[1]))]
        return mask_train, mask_val


def cat10_model_simple():
    # build model structure
    model = Sequential()
    model.add(L.Flatten(input_shape=(64, 64, 1)))
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
    model.add(L.Flatten(input_shape=(64, 64, 1)))
    model.add(L.Dense(128, activation='relu'))
    model.add(L.Dense(64, activation='relu'))
    model.add(L.Dense(40, activation='softmax'))
    # set up optimizer and compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def cat40_model_conv():
    # build model structure
    model = Sequential(name='cat40')

    model.add(L.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 1), name='cat40_1'))
    model.add(L.Conv2D(32, (3, 3), padding='same', activation='relu', name='cat40_2'))
    model.add(L.MaxPooling2D(name='cat40_3'))
    model.add(L.BatchNormalization(name='cat40_4'))

    model.add(L.Conv2D(64, (3, 3), padding='same', activation='relu', name='cat40_5'))
    model.add(L.Conv2D(64, (3, 3), padding='same', activation='relu', name='cat40_6'))
    model.add(L.MaxPooling2D(name='cat40_7'))
    model.add(L.BatchNormalization(name='cat40_8'))

    model.add(L.Flatten(name='cat40_9'))
    model.add(L.Dense(64, name='cat40_10'))
    # model.add(L.BatchNormalization())
    model.add(L.Activation('relu', name='cat40_11'))
    model.add(L.Dense(40, activation='softmax', name='cat40_12'))

    # set up optimizer and compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def cat10_model_conv():
    # build model structure
    model = Sequential(name='cat10')

    model.add(L.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 1), name='cat10_1'))
    model.add(L.Conv2D(16, (3, 3), padding='same', activation='relu', name='cat10_2'))
    model.add(L.MaxPooling2D(name='cat10_3'))
    model.add(L.BatchNormalization(name='cat10_4'))

    model.add(L.Conv2D(32, (3, 3), padding='same', activation='relu', name='cat10_5'))
    model.add(L.Conv2D(32, (3, 3), padding='same', activation='relu', name='cat10_6'))
    model.add(L.MaxPooling2D(name='cat10_7'))
    model.add(L.BatchNormalization(name='cat10_8'))

    model.add(L.Flatten(name='cat10_9'))
    model.add(L.Dense(64, name='cat10_10'))
    # model.add(L.BatchNormalization())
    model.add(L.Activation('relu', name='cat10_11'))
    model.add(L.Dense(10, activation='softmax', name='cat10_12'))

    # set up optimizer and compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model

def ensemble(model_cat10, model_cat40, num_cat):
    imgs = L.Input(shape=(64, 64, 1), name='ensemble_input')
    cat10 = model_cat10(imgs)
    cat40 = model_cat40(imgs)
    ens = L.concatenate([cat10, cat40], name='ensemble_merge')
    y = L.Dense(num_cat, activation='softmax', name='ensemble_dense')(ens)
    return Model(inputs=imgs, outputs=y, name='ensemble')


if __name__ == '__main__':
    # trainer = OracleNetTrainer()

    # model_cat10 = load_model('./model/weights_norm_cat10_v2.hdf5')
    # model_cat40 = load_model('./model/weights_norm_cat40_v2.hdf5')
    # model_cat10.name = 'cat10'
    # model_cat40.name = 'cat40'
    # model_cat10.trainable = False
    # model_cat40.trainable = False
    # # ensemble_cat10 = ensemble(model_cat10, model_cat40, 10)
    # ensemble_cat40 = ensemble(model_cat10, model_cat40, 40)
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # # ensemble_cat10.compile(loss='categorical_crossentropy', optimizer=sgd,
    # #                        metrics=['accuracy'])
    # ensemble_cat40.compile(loss='categorical_crossentropy', optimizer=sgd,
    #                        metrics=['accuracy'])
    # trainer.train(ensemble_cat40, num_cat=40, batch_size=32, epochs=30)
    # # print(trainer.error_matrix(model_cat10.predict, model_cat40.predict))

    # model = cat10_model_conv()
    # trainer.train(model, num_cat=10, epochs=30)

    # model = cat40_model_conv()
    # trainer.train(model, num_cat=40, epochs=30)

    # model = load_model('./model/weights_ensemble_cat40.hdf5')
    # trainer.evaluate(model, num_cat=40)
    # OracleNetTrainer.run_test(model, 40)
    # plt.show()

    # model = load_model('./model/weights_ensemble_cat10.hdf5')
    # model = load_model('./model/weights_norm_cat10.hdf5')
    # trainer.evaluate(model, num_cat=10)
    OracleNetTrainer.run_test(model, 10)
    plt.show()

    # model = load_model('./model/weights_conv_cat40.hdf5')
    # trainer.evaluate(model, num_cat=40, confusion=False)

    # model = load_model('./model/weights_simple_cat10.hdf5')
    # trainer.evaluate(model, num_cat=10)
    # plt.show()

    # model = cat10_model_simple()
    # trainer.train(model, num_cat=10, epochs=150)

    # model = cat40_model_simple()
    # trainer.train(model, num_cat=40, epochs=150)
