import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from utils import *

import imgaug as ia
import imgaug.augmenters as iaa

ALL_KEYS = ('val', 'flipped', 'affine', 'noise', 'test')


class Dataset:
    # default resize images to following size
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    @staticmethod
    def aug_filter(fname, keys=None):
        if keys is not None:
            for key in keys:
                if key in fname:
                    return False
        return True

    @staticmethod
    def folder_to_cat(folder, num_cat=10):
        """ Convert the folder name into categories
        """
        assert num_cat in [10, 40], "Number of categories can only be 10 or 40"
        return (int(folder) - 102) // (40 / num_cat)

    @staticmethod
    def cat40_to_cat10(cats):
        return cats // 4

    @staticmethod
    def get_image_file(fname):
        img = Image.open(fname).convert('L')
        return img

    @staticmethod
    def get_image_folder(folder, num_cat=10, one_hot=False, filter_keys=None, inclusive=False, names=False, padding=0, normalize=False):
        """
        Get all resized images and their labels in one folder
        :param folder: folder name
        :param num_cat: number of categories, 10 or 40
        :param one_hot: whether to use one hot encoding
        :param padding: add white padding around the image
        :return: images, labels
        """
        if filter_keys is not None or one_hot:
            path = os.path.join('../dataset', folder)
        else:
            path = folder
        fnames = list(filter(lambda x: Dataset.aug_filter(x, filter_keys) ^ inclusive, os.listdir(path)))
        imgs = []
        for fname in fnames:
            img = Dataset.get_image_file(os.path.join(path, fname))
            img = img.resize((Dataset.IMG_HEIGHT, Dataset.IMG_WIDTH), Image.ANTIALIAS)
            if normalize:
                img = np.array(img)
                img = (img - img.min()) / (img.max() - img.min())
            if padding > 0:
                img = ImageOps.expand(img, (padding, padding, padding, padding), 255)
            imgs.append(np.array(img))
        imgs = np.array(imgs)
        if len(imgs.shape) < 4:
            imgs = imgs.reshape(imgs.shape + (1,))

        if names:
            return imgs, fnames

        cat = Dataset.folder_to_cat(folder, num_cat)
        cats = np.repeat(cat, imgs.shape[0])
        if one_hot:
            cats = to_categorical(cats, num_cat)
        return imgs, cats

    @staticmethod
    def to_onehot(cats, num_cat=10):
        return to_categorical(cats, num_cat)

    @staticmethod
    def load_data(num_cat=10, one_hot=False, filter_keys=None, inclusive=False, padding=0, normalize=False):
        """
        Load all images in dataset folder
        :param num_cat: number of categories
        :param one_hot: whether to use one hot encoding
        :return: X: shape [N, IMG_HEIGHT, IMG_WIDTH],
                 y: shape [N,] or [N, num_cat]
        """
        dirs = os.listdir('../dataset/')
        X = None
        y = []
        for dir in dirs:
            _X, _y = Dataset.get_image_folder(folder=dir, num_cat=num_cat, filter_keys=filter_keys, inclusive=inclusive,
                                              padding=padding, normalize=normalize)
            if X is None:
                X = _X
            else:
                X = np.concatenate([X, _X])
            y = np.concatenate([y, _y])
        if one_hot:
            y = Dataset.to_onehot(y, num_cat)
        return X, y

    @staticmethod
    def clear_folder(folder, path='../dataset/', filter_keys=None):
        """

        :param folder:
        :param path:
        :param filter_keys: inclusive, remove files with keys
        :return:
        """
        path = path + folder + '/'
        fnames = list(filter(lambda x: not Dataset.aug_filter(x, filter_keys), os.listdir(path)))
        for fname in fnames:
            os.remove(path + fname)

    @staticmethod
    def clear_all(path='../dataset/', filter_keys=None):
        if filter_keys is None:
            filter_keys = set(ALL_KEYS) - {'val', 'test'}
        for folder in os.listdir(path):
            Dataset.clear_folder(folder, path, filter_keys=filter_keys)


class DataAugmentation(object):
    def __init__(self):
        self.flip_aug = iaa.Fliplr(1.0)
        self.affine_aug = iaa.Affine(
            scale={'x': (0.90, 1.10), 'y': (0.90, 1.10)},
            translate_px={'x': (-10, 10), 'y': (-10, 10)},
            rotate=(-10, 10),
            shear=(-10, 10),
            mode='constant',
            cval=(255, 255)
        )
        self.noise_aug = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_px={"x": (-5, 5), "y": (-5, 5)},
                mode='constant',
                cval=(255, 255)
            ),
            iaa.Sequential([
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.ContrastNormalization((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
            ], random_order=True)
        ], random_order=False)

    def flip(self, imgs):
        return self.flip_aug(images=imgs)

    def affine(self, imgs):
        return self.affine_aug(images=imgs)

    def noise(self, imgs):
        return self.noise_aug(images=imgs)

    @staticmethod
    def aug_folder(method, label, folder_src, folder_dst=None):
        if folder_dst is None:
            folder_dst = folder_src
        path_src = '../dataset/' + folder_src + '/'
        path_dst = '../dataset/' + folder_dst + '/'
        imgs, fnames = Dataset.get_image_folder(folder_src, filter_keys=ALL_KEYS, names=True)
        imgs = method(imgs)
        for i in range(len(imgs)):
            prefix = fnames[i].split('.')[0]
            Image.fromarray(imgs[i]).save(path_dst + prefix + '_' + label + '.jpg')

    @staticmethod
    def mark_val_folder(folder, path='../dataset/', val_per_cat=10, filter_keys=None):
        """
        Mark validation samples.
        :param folder:
        :param path:
        :param val_size:
        :param filter_keys: exclusive, do not select files with keys
        :return:
        """
        path = path + folder + '/'
        if filter_keys is None:
            filter_keys = ALL_KEYS
        fnames = list(filter(lambda x: Dataset.aug_filter(x, filter_keys), os.listdir(path)))
        fnames = np.random.choice(fnames, val_per_cat, replace=False)
        for fname in fnames:
            prefix = fname.split('.')[0]
            os.rename(path + fname, path + prefix + '_val.jpg')

    @staticmethod
    def reset_val_folder(folder, path='../dataset/'):
        path = path + folder + '/'
        fnames = list(filter(lambda x: not Dataset.aug_filter(x, ['val']), os.listdir(path)))
        for fname in fnames:
            # danfiaisd_val.jpg
            prefix = fname.split('.')[0][:-4]
            os.rename(path + fname, path + prefix + '.jpg')

    @staticmethod
    def mark_val_all(path='../dataset/', val_per_cat=10, filter_keys=None):
        for folder in os.listdir(path):
            DataAugmentation.mark_val_folder(folder, path, val_per_cat=val_per_cat, filter_keys=filter_keys)

    @staticmethod
    def reset_val_all(path='../dataset/'):
        for folder in os.listdir(path):
            DataAugmentation.reset_val_folder(folder, path)

    @staticmethod
    def preview(method, fname):
        img = np.array(Dataset.get_image_file(fname))
        new_img = method([img])[0]
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(new_img)
        plt.show()


def pipeline():
    """ Run the pipeline to construct entire dataset.
    """
    # set/reset valiation set
    DataAugmentation.mark_val_all()
    # DataAugmentation.reset_val_all()

    # augmentation
    auger = DataAugmentation()
    # local flip horizontal
    auger.aug_folder(auger.flip, 'flipped', '0114')
    auger.aug_folder(auger.flip, 'flipped', '0115')
    auger.aug_folder(auger.flip, 'flipped', '0116')
    auger.aug_folder(auger.flip, 'flipped', '0117')
    # cross flip horizontal
    auger.aug_folder(auger.flip, 'flipped', '0128', '0127')
    auger.aug_folder(auger.flip, 'flipped', '0127', '0128')
    auger.aug_folder(auger.flip, 'flipped', '0119', '0121')
    auger.aug_folder(auger.flip, 'flipped', '0121', '0119')
    auger.aug_folder(auger.flip, 'flipped', '0130', '0131')
    auger.aug_folder(auger.flip, 'flipped', '0131', '0130')
    # local affine to balance data
    for i in range(102, 142):
        auger.aug_folder(auger.noise, 'noise', '0{}'.format(i))
        if i not in [114, 115, 116, 117, 127, 128, 119, 121, 130, 131]:
            auger.aug_folder(auger.affine, 'affine', '0{}'.format(i))
    # Dataset.clear_all()


if __name__ == '__main__':
    # run pipelined augmentation
    Dataset.clear_all()
    pipeline()
