import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset, pipeline, ALL_KEYS
from PIL import Image
import pickle
import os
from scipy.special import softmax

from utils import plot_confusion_matrix

Dataset.IMG_HEIGHT = 64
Dataset.IMG_WIDTH = 64


# PADDING = 40
# SCALE_THRESHOLD = 0.99

class TemplateMatch:
    def __init__(self, padding=40, scale_threshold=0.99, method='CCORR_NORMED'):
        self.padding = padding
        self.scale_threshold = scale_threshold
        self.templates = []
        self.method = method
        self.white_thres = 150

    def get_best_scale(self, img):
        # set near white color to white
        img[img > self.white_thres] = 255
        # plt.hist(img.ravel())
        # plt.show()
        # add padding
        img = cv2.copyMakeBorder(img, self.padding, self.padding, self.padding, self.padding,
                                 cv2.BORDER_CONSTANT, None, 255)

        # find the center of mass
        accx = np.array([np.arange(img.shape[0])]).T
        accy = np.arange(img.shape[1])
        inv = 255 - img
        suminv = np.sum(inv)
        cx = int(round(np.sum(inv * accx) / suminv))
        cy = int(round(np.sum(inv * accy) / suminv))
        # print(cx,cy)

        # find the smallest square that enclose 99% of mass
        thres = self.scale_threshold
        halfsize = 10
        while 1:
            tmp = inv[cx - halfsize:cx + halfsize + 1, cy - halfsize:cy + halfsize + 1]
            if np.sum(tmp) > thres * suminv:
                break
            halfsize += 1

        return img[cx - halfsize:cx + halfsize + 1, cy - halfsize:cy + halfsize + 1]

    def get_template(self, group, size=(80, 80)):
        """
        Generate template for one category.
        :param group: a group of images with the same category
        :param size: the size of the template
        :return: template image
        """
        templ = np.zeros(size, np.float)
        for i in range(group.shape[0]):
            img = self.get_best_scale(group[i])
            templ += cv2.resize(img, size)
        # show template
        # plt.imshow(templ / group.shape[0])
        # plt.show()
        return templ / group.shape[0]

    def score(self, group, template):
        assert self.method in ['MSE', 'MSE_NORMED', 'CCORR', 'CCORR_NORMED']

        # get best scale
        new_group = np.zeros((group.shape[0], *template.shape), np.uint8)
        for i in range(group.shape[0]):
            new_group[i] = cv2.resize(self.get_best_scale(group[i]), template.shape)

        # custom implement
        if self.method == 'MSE':
            return np.mean((new_group - template) ** 2, axis=(1, 2))
        if self.method == 'MSE_NORMED':
            num = np.mean((new_group - template) ** 2, axis=(1, 2))
            den = np.sqrt(np.sum(new_group ** 2, axis=(1, 2)) * np.sum(template ** 2))
            return num / den
        if self.method == 'CCORR':
            return np.sum(new_group * template, axis=(1, 2))
        if self.method == 'CCORR_NORMED':
            num = np.sum(new_group * template, axis=(1, 2))
            den = np.sqrt(np.sum(new_group ** 2, axis=(1, 2)) * np.sum(template ** 2))
            return num / den

        # use cv2.matchTemplate
        # else:
        # # move template around to get the best match
        # res = [cv2.matchTemplate(new_group[i], template.astype(np.uint8), eval('cv2.TM_' + method)) for i in
        #        range(new_group.shape[0])]
        # minmax = np.array([cv2.minMaxLoc(res[i]) for i in range(new_group.shape[0])])
        # min_vals = minmax[:, 0]
        # max_vals = minmax[:, 1]
        # if method in ['SQDIFF', 'SQDIFF_NORMED']:
        #     return min_vals
        # else:
        #     return max_vals

    def match(self, dataset):
        scores = np.array([self.score(dataset, t) for t in self.templates])
        if self.method in ['MSE', 'MSE_NORMED']:
            return scores.argmin(axis=0)
        else:
            return scores.argmax(axis=0)

    def train(self, padding, scale_thres, output_dir):
        self.padding = padding
        self.scale_threshold = scale_thres
        self.templates = []

        X_train, y_train = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=ALL_KEYS, inclusive=False)

        for i in range(40):
            templ = self.get_template(X_train[y_train == i])
            self.templates.append(templ)
            # plt.imshow(temp,cmap='gray')
            # plt.show()
        if output_dir:
            model = {'padding': self.padding, 'scale_thres': self.scale_threshold, 'templates': self.templates}
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            f = open(os.path.join(output_dir, 'templ.pkl'), 'wb')
            pickle.dump(model, f, protocol=4)
            f.close()

    def load_model(self, model_dir):
        f = open(model_dir, 'rb')
        param = pickle.load(f)
        f.close()
        self.padding = param['padding']
        self.scale_threshold = param['scale_thres']
        self.templates = param['templates']

    def evaluate(self, method='', show_plot=False):
        if method:
            self.method = method

        X_val, y_val = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['test'], inclusive=True,
                                         normalize=True)
        X_val *= 255

        m = self.match(X_val.copy())

        print('method:', self.method)
        print('cat40', np.mean(m == y_val))
        print('cat10', np.mean(m // 4 == y_val // 4))

        if show_plot:
            plot_confusion_matrix(y_val // 4, m // 4, 10)
            plt.show()
            plot_confusion_matrix(y_val, m, 40)
            plt.show()

    def __call__(self, X):
        '''
        :param X: nparray m*h*w
        :return: softmax score of each category
        '''
        if X.ndim == 4:
            X = X[:, :, :, 0]
        scores = np.array([self.score(X, t) for t in self.templates])
        return softmax(scores.T, axis=1)


if __name__ == '__main__':
    methods = ['MSE', 'MSE_NORMED', 'CCORR', 'CCORR_NORMED']

    tm = TemplateMatch()
    # tm.train(40, 0.99, output_dir='./model/')
    tm.load_model('./model/templ.pkl')

    for method in ['CCORR_NORMED']:
        tm.evaluate(method)
