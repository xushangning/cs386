import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset, pipeline, ALL_KEYS
from PIL import Image
import pickle
import os

from utils import plot_confusion_matrix

Dataset.IMG_HEIGHT = 89
Dataset.IMG_WIDTH = 81

PADDING = 40
SCALE_THRESHOLD = 0.99


def get_best_scale(img):
    # find the center of mass
    accx = np.array([np.arange(img.shape[0])]).T
    accy = np.arange(img.shape[1])
    inv = 255 - img
    suminv = np.sum(inv)
    cx = int(round(np.sum(inv * accx) / suminv))
    cy = int(round(np.sum(inv * accy) / suminv))
    # print(cx,cy)

    # find the smallest square that enclose 99% of mass
    thres = SCALE_THRESHOLD
    halfsize = 30
    while 1:
        tmp = inv[cx - halfsize:cx + halfsize + 1, cy - halfsize:cy + halfsize + 1]
        if np.sum(tmp) > thres * suminv:
            break
        halfsize += 1

    return img[cx - halfsize:cx + halfsize + 1, cy - halfsize:cy + halfsize + 1]


def get_template(group, size=(80, 80)):
    """
    Generate template for one category.
    :param group: a group of images with the same category
    :param size: the size of the template
    :return: template image
    """
    templ = np.zeros(size, np.float)
    for i in range(group.shape[0]):
        img = get_best_scale(group[i])
        templ += cv2.resize(img, size)
    # show template
    # plt.imshow(templ / group.shape[0])
    # plt.show()
    return templ / group.shape[0]


# mse error
def score(group, template, method):
    assert method in ['MSE', 'SQDIFF', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED']

    # get best scale
    new_group = np.zeros((group.shape[0], *template.shape), np.uint8)
    for i in range(group.shape[0]):
        new_group[i] = cv2.resize(get_best_scale(group[i]), template.shape)

    # custom implement
    if method == 'MSE':
        return np.array([np.mean((new_group[i] - template) ** 2) for i in range(new_group.shape[0])])
        # return np.mean((group - template) ** 2, axis=(1, 2))

    # use cv2.matchTemplate
    else:
        # move template around to get the best match
        res = [cv2.matchTemplate(new_group[i], template.astype(np.uint8), eval('cv2.TM_' + method)) for i in
               range(new_group.shape[0])]
        minmax = np.array([cv2.minMaxLoc(res[i]) for i in range(new_group.shape[0])])
        min_vals = minmax[:, 0]
        max_vals = minmax[:, 1]
        if method in ['SQDIFF', 'SQDIFF_NORMED']:
            return min_vals
        else:
            return max_vals


def match(dataset, templates, method):
    scores = np.array([score(dataset, t, method) for t in templates])
    if method in ['MSE', 'SQDIFF', 'SQDIFF_NORMED']:
        return scores.argmin(axis=0)
    else:
        return scores.argmax(axis=0)


def train(padding, scale_thres, output_dir):
    templates = []
    PADDING = padding
    SCALE_THRESHOLD = scale_thres
    X_train, y_train = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=ALL_KEYS, inclusive=False,
                                         padding=PADDING)
    for i in range(40):
        templ = get_template(X_train[y_train == i])
        templates.append(templ)
        # plt.imshow(temp,cmap='gray')
        # plt.show()
    if output_dir:
        model = {'padding': PADDING, 'scale_thres': SCALE_THRESHOLD, 'templates': templates}
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        f = open(os.path.join(output_dir, 'templ.pkl'), 'wb')
        pickle.dump(model, f, protocol=4)
        f.close()
    return model


def load_model(templ_dir):
    f = open(os.path.join(templ_dir, 'templ.pkl'), 'rb')
    param = pickle.load(f)
    f.close()
    PADDING = param['padding']
    SCALE_THRESHOLD = param['scale_thres']
    return PADDING, SCALE_THRESHOLD, param['templates']


def evaluate(templates, method, show_plot=False):
    X_val, y_val = Dataset.load_data(num_cat=40, one_hot=False, filter_keys=['val'], inclusive=True, padding=PADDING)

    m = match(X_val, templates, method)

    print('method:', method)
    print('cat40', np.mean(m == y_val))
    print('cat10', np.mean(m // 4 == y_val // 4))

    if show_plot:
        plot_confusion_matrix(y_val // 4, m // 4, 10)
        plt.show()
        plot_confusion_matrix(y_val, m, 40)
        plt.show()


if __name__ == '__main__':
    methods = ['MSE', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED']
    train(40, 0.99, output_dir='./model/')
    _, _, templates = load_model('./model/')
    for method in methods:
        evaluate(templates, method)
