import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import imgaug as ia
import imgaug.augmenters as iaa

# default resize images to following size
# IMG_HEIGHT = 89
# IMG_WIDTH = 81
IMG_HEIGHT = 64
IMG_WIDTH = 64


class Dataset:
    @staticmethod
    def aug_filter(fname, keys=None):
        if keys is not None:
            for key in keys:
                if key in fname:
                    return False
        return True

    # convert the folder name into categories
    @staticmethod
    def folder_to_cat(folder, num_cat=10):
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        return (int(folder) - 102) // (40 / num_cat)

    @staticmethod
    def cat40_to_cat10(cats):
        return cats // 4

    # read single image
    @staticmethod
    def get_image_file(fname):
        # img = np.array(Image.open(fname))
        img = Image.open(fname)
        return img #.reshape(img.shape + (1,))

    @staticmethod
    def get_image_folder(folder, num_cat=10, one_hot=False, filter_keys=None, names=False):
        """
        get all resized images and their labels in one folder
        :param folder: folder name
        :param num_cat: number of categories, 10 or 40
        :param one_hot: whether to use one hot encoding
        :return: images, labels
        """
        path = '../dataset/' + folder + '/'
        fnames = list(filter(lambda x: Dataset.aug_filter(x, filter_keys), os.listdir(path)))
        cat = Dataset.folder_to_cat(folder, num_cat)
        imgs = []
        for fname in fnames:
            img = Dataset.get_image_file(path + fname)
            img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
            imgs.append(np.array(img))
        if names:
            return imgs, fnames
        imgs = np.array(imgs)
        cats = np.repeat(cat, imgs.shape[0])
        if one_hot:
            cats = to_categorical(cats, num_cat)
        return imgs, cats

    @staticmethod
    def to_onehot(cats, num_cat=10):
        return to_categorical(cats, num_cat)

    @staticmethod
    def load_data(num_cat=10, one_hot=False, filter_keys=None):
        """
        load all images in dataset folder
        :param num_cat: number of categories
        :param one_hot: whether to use one hot encoding
        :return: X: shape [N, IMG_HEIGHT, IMG_WIDTH],
                 y: shape [N,] or [N, num_cat]
        """
        dirs = os.listdir('../dataset/')
        X = None
        y = []
        for dir in dirs:
            _X, _y = Dataset.get_image_folder(dir, num_cat, filter_keys)
            if X is None:
                X = _X
            else:
                X = np.concatenate([X, _X])
            y = np.concatenate([y, _y])
        if one_hot:
            y = Dataset.to_onehot(y, num_cat)
        return X, y


class DataAugmentation(object):
    def __init__(self):
        self.flip_aug = iaa.Fliplr(1.0)
        self.slight_affine_aug = iaa.Affine(
            scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
            translate_px={'x': (-5, 5), 'y': (-5, 5)},
            rotate=(-5, 5),
            shear=(-3, 3),
            mode='constant',
            cval=(255, 255)
        )

    def flip(self, imgs):
        return self.flip_aug(images=imgs)

    def slight_affine(self, imgs):
        return self.slight_affine_aug(images=imgs)

    def flip_folder(self, folder_src, folder_dst):
        path_src = '../dataset/' + folder_src + '/'
        path_dst = '../dataset/' + folder_dst + '/'
        imgs, fnames = Dataset.get_image_folder(folder_src, filter_keys=['val', 'flipped'], names=True)
        imgs = self.flip(imgs)
        for i in range(len(imgs)):
            prefix = fnames[i].split('.')[0]
            Image.fromarray(imgs[i]).save(path_dst + prefix + '_flipped.jpg')

    @staticmethod
    def preview(method, fname):
        img = Dataset.get_image_file(fname)
        new_img = method([img])[0]
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(new_img)
        plt.show()


if __name__ == '__main__':
    auger = DataAugmentation()
    # auger.flip_folder('0130', '0131')
    auger.flip_folder('0131', '0130')

    # auger.preview(auger.flip, '../dataset/0130/person_0000.jpg')
    # auger.preview(auger.slight_affine, '../dataset/0130/person_0000.jpg')

    # X, y = Dataset.load_data()
    # print(X.shape)
    # print(y.shape)

    # shapes = np.array([x.shape for x in X])
    # print(np.unique(shapes[:, 0]))
    # print(np.unique(shapes[:, 1]))
    # [89 93 101]
    # [81 91]
    pass
