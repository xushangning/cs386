import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import sys


def dct_tile(img, i, j, tile, channel=0, rate=2, divide_ref=True):
    img = np.float32(img)
    if len(img.shape) < 3:
        img_tile = img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
    else:
        img_tile = img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]
    img_ref = cv2.resize(down_sample(img_tile, rate), dsize=img_tile.shape, interpolation=cv2.INTER_AREA)
    if divide_ref:
        tmp = cv2.dct(img_tile) / cv2.dct(img_ref)
    else:
        tmp = (cv2.dct(img_tile), cv2.dct(img_ref))
    return tmp


def self_ref_dct_complete(img, tile, channel=0, rate=2, threshold=20):
    if len(img.shape) > 3:
        img = img[:, :, channel]
    nx = img.shape[0] // tile
    ny = img.shape[1] // tile
    dcts = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            dct = dct_tile(img[:, :], i, j, tile, rate=rate).flatten()
            dct = dct[abs(dct) < threshold]
            dcts.append(dct)
    return np.array(dcts)


def dct_feature_extract(img, tile, channel=0, samples=50, ref_rate=2, threshold=20, div_dct=True):
    if len(img.shape) > 3:
        img = img[:, :, channel]

    xs = list(range(img.shape[0] // tile))
    ys = list(range(img.shape[1] // tile))
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    idxs = list(zip(xs, ys))[:samples]

    abs_dev = lambda x: abs(x).mean()
    sqr_dev = lambda x: (x ** 2).mean() ** 0.5

    dcts = []
    abds = []
    stds = []
    for i, j in idxs:
        if div_dct:
            dct = dct_tile(img[:, :], i, j, tile, rate=ref_rate, divide_ref=div_dct).flatten()
            dct = dct[abs(dct) < threshold]
            abd = abs_dev(dct)
            std = sqr_dev(dct)
        else:
            dct, ref = dct_tile(img[:, :], i, j, tile, rate=ref_rate, divide_ref=div_dct)
            abd = abs_dev(dct) / abs_dev(ref)
            std = sqr_dev(dct) / sqr_dev(ref)
        dcts.append(dct)
        abds.append(abd)
        stds.append(std)
    return np.array([np.mean(abds), np.min(abds), np.max(abds),
                     np.mean(stds), np.min(stds), np.max(stds)]), \
           np.array(dcts), np.array(abds), np.array(stds)


def extract_feature_folder(folder, tile, channel=0, samples=50, ref_rate=2, threshold=20, div_dct=True):
    fnames = [e for e in os.listdir(folder) if e.split('.')[-1] in ('bmp', 'jpg', 'png')]
    feats = []
    for i, fname in enumerate(fnames):
        print('{} / {}'.format(i, len(fnames)))
        img = get_image(folder + fname)
        feat, _, _, _ = dct_feature_extract(img, tile, channel, samples, ref_rate, threshold, div_dct)
        feats.append(feat)
    return np.array(feats)


# def tile_dct(img, i, j, tile, channel=0):
#     if len(img.shape) < 3:
#         return cv2.dct(np.float32(img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]))
#     elif len(img.shape) < 4:
#         if img.shape[2] <= 3:
#             return cv2.dct(np.float32(img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]))
#         else:
#             return cv2.dct(np.float32(img[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]))
#     else:
#         return cv2.dct(np.float32(img[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]))
#
#
# def dct_hist_union(img, tile, channel=0):
#     nx = img.shape[0] // tile
#     ny = img.shape[1] // tile
#     dcts = np.array([])
#     for i in range(nx):
#         for j in range(ny):
#             dcts = np.concatenate([dcts, tile_dct(img, i, j, tile, channel).flatten()])
#     hst = plt.hist(dcts)[0]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify the image folder.")
        exit(-1)
    folder = sys.argv[1]
    tile = 32
    channel = 0
    samples = 50
    ref_rate = 2
    threshold = 20
    div_dct = True
    if len(sys.argv) > 2:
        tile = int(sys.argv[2])
    if len(sys.argv) > 3:
        channel = int(sys.argv[3])
    if len(sys.argv) > 4:
        samples = int(sys.argv[4])
    if len(sys.argv) > 5:
        ref_rate = int(sys.argv[5])
    if len(sys.argv) > 6:
        threshold = int(sys.argv[6])
    if len(sys.argv) > 7:
        div_dct = bool(sys.argv[7])

    feats = extract_feature_folder(folder, tile, channel, samples, ref_rate, threshold, div_dct)
    np.savetxt(folder + "tile_{}_channel_{}_samples_{}_rate_{}_threshold_{}_div_{}.txt".format(tile, channel, samples,
                                                                                               ref_rate, threshold,
                                                                                               div_dct), feats)

    # plt.figure()
    # img_4k = get_image("./images/4k/1.bmp")
    # dct_hist_union(img_4k, 32)
    # plt.figure()
    # img_18bc = get_image("./images/1080P/bicubic/1.bmp")
    # dct_hist_union(img_18bc, 32)
    # plt.show()
    pass
