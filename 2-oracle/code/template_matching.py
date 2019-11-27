import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from PIL import Image

Dataset.IMG_HEIGHT = 89
Dataset.IMG_WIDTH = 81


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
    thres = 0.99
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
    new_group = np.zeros((group.shape[0],*template.shape),np.uint8)
    for i in range(group.shape[0]):
        new_group[i] = cv2.resize(get_best_scale(group[i]),template.shape)

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


def main():
    templates = []
    groups = []
    methods = ['MSE', 'SQDIFF', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED']
    for i in range(40):
        g, _ = Dataset.get_image_folder('0' + str(102 + i), 40, padding=40)
        groups.append(g)
        temp = get_template(g)
        templates.append(temp)
        # plt.imshow(temp,cmap='gray')
        # plt.show()

    for method in ['MSE']:
        print(method)

        hits40 = []
        hits10 = []
        total = []
        for i in range(40):
            m = match(groups[i], templates, method)
            hits40.append(np.sum(m == i))
            hits10.append(np.sum(m // 4 == i // 4))
            total.append(len(m))
            print(i, np.mean(m == i))

        print('cat40', 1.0 * sum(hits40) / sum(total))
        print('cat10', 1.0 * sum(hits10) / sum(total))


if __name__ == '__main__':
    main()
    # g, _ = Dataset.get_image_folder('0' + str(102), 40, padding=40)
    # get_template(g)
    # # plt.imshow(g[0])
    # # plt.show()
    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # get_best_scale(a,(3,2))
    # print(a*np.array([[1,2,3]]).T)
