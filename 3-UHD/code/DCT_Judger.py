import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def tile_dct(img, i, j, tile, channel=0):
    if len(img.shape) < 3:
        return cv2.dct(np.float32(img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]))
    elif len(img.shape) < 4:
        if img.shape[2] <= 3:
            return cv2.dct(np.float32(img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]))
        else:
            return cv2.dct(np.float32(img[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]))
    else:
        return cv2.dct(np.float32(img[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]))


def dct_hist_union(img, tile, channel=0):
    nx = img.shape[0] // tile
    ny = img.shape[1] // tile
    dcts = np.array([])
    for i in range(nx):
        for j in range(ny):
            dcts = np.concatenate([dcts, tile_dct(img, i, j, tile, channel).flatten()])
    hst = plt.hist(dcts)[0]


if __name__ == '__main__':
    plt.figure()
    img_4k = get_image("./images/4k/1.bmp")
    dct_hist_union(img_4k, 32)
    plt.figure()
    img_18bc = get_image("./images/1080P/bicubic/1.bmp")
    dct_hist_union(img_18bc, 32)
    plt.show()
    pass
