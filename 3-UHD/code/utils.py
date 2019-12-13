import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_image(fname):
    return cv2.imread(fname)


def get_folder(folder="./images/4k/", callback=None, num_img=None):
    fnames = [e for e in os.listdir(folder) if e.split('.')[-1] in ('bmp', 'jpg', 'png')]
    imgs = []
    for i, fname in enumerate(fnames):
        if num_img is not None and i >= num_img:
            break
        if callback is not None:
            callback()
        print('{} / {}'.format(i, len(fnames)))
        img = get_image(folder + fname)
        imgs.append(img)
    return np.array(imgs)


def down_sample(img, rate=2, sx=0, sy=0):
    if len(img.shape) < 3:
        return img[sx::rate, sy::rate]
    else:
        return img[sx::rate, sy::rate, :]


def vis_hist(img, thresholds=(100, 1000, 10000), level=None):
    tmp = img.flatten()

    if thresholds is None:
        plt.hist(tmp, bins=100)
    else:
        num_plots = len(thresholds) + 1

        plt.figure(figsize=[12, 4 * ((num_plots + 1) // 2)])

        for idx in range(num_plots):
            if level == idx:
                break
            plt.subplot((num_plots + 1) // 2, 2, idx + 1)
            if idx == 0:
                plt.hist(tmp[abs(tmp) < thresholds[0]], bins=100)
            elif idx == num_plots - 1:
                plt.hist(tmp[abs(tmp) >= thresholds[idx - 1]], bins=100)
            else:
                plt.hist(tmp[(abs(tmp) >= thresholds[idx - 1]) & (abs(tmp) < thresholds[idx])], bins=100)

    plt.show()


if __name__ == '__main__':
    imgs = get_folder()
    print(imgs.shape)
