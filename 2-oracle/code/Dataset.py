import os
from PIL import Image

import numpy as np
from keras.utils import to_categorical

# resize images to following size
IMG_HEIGHT = 89
IMG_WIDTH = 81


class Dataset:
    @staticmethod
    def folder_to_cat(folder, num_cat=10):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        return (int(folder) - 102) // (40 / num_cat)

    @staticmethod
    def get_image_file(fname):
        img = Image.open(fname)
        return np.array(img)

    @staticmethod
    def get_image_folder(folder, num_cat=10, one_hot=False):
        path = '../dataset/' + folder + '/'
        fnames = os.listdir(path)
        cat = Dataset.folder_to_cat(folder, num_cat)
        imgs = []
        for fname in fnames:
            img = Dataset.get_image_file(path + fname)
            # print(img.shape)
            imgs.append(img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS))
        imgs = np.array(imgs)
        cats = np.repeat(cat, imgs.shape[0])
        if one_hot:
            cats = to_categorical(cats, num_cat)
        return imgs, cats

    @staticmethod
    def to_onehot(cats, num_cat=10):
        return to_categorical(cats, num_cat)

    @staticmethod
    def load_data(num_cat=10, one_hot=False):
        dirs = os.listdir('../dataset/')
        X = []
        y = []
        for dir in dirs:
            _X, _y = Dataset.get_image_folder(dir, num_cat)
            # print(_X.shape)
            X = np.concatenate([X, _X])
            y = np.concatenate([y, _y])
        if one_hot:
            y = Dataset.to_onehot(y, num_cat)
        return X, y


if __name__ == '__main__':
    X, y = Dataset.load_data()
    print(X.shape)
    print(y.shape)

    # shapes = np.array([x.shape for x in X])
    # print(np.unique(shapes[:, 0]))
    # print(np.unique(shapes[:, 1]))
    # [89 93 101]
    # [81 91]
    pass
