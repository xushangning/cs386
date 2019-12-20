import cv2
import numpy as np
from utils import *
import os
import argparse
import pandas as pd


def dct_tile(img, i, j, tile, channel=0, rate=2, divide_ref=True, ref_int=cv2.INTER_AREA, origin=None):
    img = np.float32(img)
    if len(img.shape) < 3:
        img_tile = img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
    else:
        img_tile = img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, channel]
    if ref_int is not None:
        img_ref = cv2.resize(down_sample(img_tile, rate), dsize=img_tile.shape, interpolation=ref_int)
        if origin is not None:
            tmp = origin / cv2.dct(img_ref)
        else:
            if divide_ref:
                tmp = cv2.dct(img_tile) / cv2.dct(img_ref)
            else:
                tmp = (cv2.dct(img_tile), cv2.dct(img_ref))
    else:
        tmp = cv2.dct(img_tile)
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


def dct_feature_extract(img, tile, channel=0, samples=50, ref_rate=2, threshold=20,
                        div_dct=True, offset=0, ref_method='AR'):
    ref_method_dict = {
        'AR': cv2.INTER_AREA,
        'NN': cv2.INTER_NEAREST,
        'BL': cv2.INTER_LINEAR,
        'BC': cv2.INTER_CUBIC,
        'NO': None
    }
    assert ref_method in ref_method_dict.keys(), \
        'The reference method can only be one of ' + str(ref_method_dict.keys())
    if ref_method == 'NO':
        offset = 0
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
            dct = dct_tile(img[:, :], i, j, tile, rate=ref_rate, divide_ref=div_dct,
                           ref_int=ref_method_dict[ref_method]).flatten() - offset
            dct = dct[abs(dct) < threshold]
            abd = abs_dev(dct)
            std = sqr_dev(dct)
        else:
            dct, ref = dct_tile(img[:, :], i, j, tile, rate=ref_rate, divide_ref=div_dct)
            abd = abs_dev(dct)  # / abs_dev(ref)
            std = sqr_dev(dct)  # / sqr_dev(ref)
        dcts.append(dct)
        abds.append(abd)
        stds.append(std)
    return np.array([np.nanmean(abds), np.nanmin(abds), np.nanmax(abds),
                     np.nanmean(stds), np.nanmin(stds), np.nanmax(stds)]), \
           np.array(dcts), np.array(abds), np.array(stds)


def extract_feature_folder(folder, tile, channel=0, samples=50, ref_rate=2, threshold=20,
                           div_dct=True, return_fnames=False, size=None, offset=0, ref_method='AR'):
    fnames = [e for e in os.listdir(folder) if e.split('.')[-1] in ('bmp', 'jpg', 'png')]
    print("Processing {}".format(folder))
    feats = []
    if size is None:
        size = len(fnames)
    for i, fname in enumerate(fnames):
        if i >= size:
            break
        print('{} / {}'.format(i, min(len(fnames), size)))
        img = get_image(os.path.join(folder, fname))
        feat, _, _, _ = dct_feature_extract(img, tile, channel, samples,
                                            ref_rate, threshold, div_dct, offset, ref_method)
        feats.append(feat)

    if return_fnames:
        return np.array(feats), fnames
    return np.array(feats)


def extract_feature_single(fname, tile, channel=0, samples=50, ref_rate=2, threshold=20,
                           div_dct=True, offset=1, ref_method='AR'):
    img = get_image(fname)
    feat, _, _, _ = dct_feature_extract(img, tile, channel, samples, ref_rate, threshold, div_dct, offset, ref_method)
    print(np.array(feat))


def classify_folder(folder, output_file, tile, channel=0, samples=50, ref_rate=2, ref_method='AR', threshold=20,
                    method='L1', size=None, offset=1):
    assert method in ['L1', 'L2', 'both'], "Parameter 'method' should be one of 'L1', 'L2', 'both'."
    thresholds_dict = {
        2:{
            'AR': (2.1545941829681396, 3.656043997245576),
            'NN': (2.1711327396333218, 3.666123659446196),
            'BL': (4.140497922897339, 6.317052318760898),
            'BC': (4.093102689832449, 6.328407565850434),
        },
        3:{
            'AR': (3.1917893290519714, 5.072529512846611),
        }
    }
    if ref_rate not in thresholds_dict.keys() or ref_method not in thresholds_dict[ref_rate].keys():
        print('Sorry! We do not support ref_rate={}, ref_method={}'.format(ref_rate, ref_method))
        return

    feats, fnames = extract_feature_folder(folder, tile, channel, samples, ref_rate, threshold,
                                           return_fnames=True, size=size, offset=offset, ref_method=ref_method)

    thresholds_cut = thresholds_dict[ref_rate][ref_method]
    mask_L1 = (feats[:, 0] > thresholds_cut[0])
    mask_L2 = (feats[:, 3] > thresholds_cut[1])
    if method == 'L1':
        res = mask_L1
    elif method == 'L2':
        res = mask_L2
    elif method == 'both':
        res = mask_L1 & mask_L2

    if 'csv' in output_file:
        pd.DataFrame({'fname': fnames, 'label': res}).to_csv(output_file, index=False)
    else:
        np.savetxt(output_file, res)


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Simple Thresholding Classifier")
    parser.add_argument("--input_folder", help="Input image folder.", type=str)
    parser.add_argument(
        "--output_filename", help="The output txt file of classification results.",
        default="output.txt", type=str
    )
    parser.add_argument(
        "--max_image", help="Divide reference before statistics.",
        default=None, type=int)
    parser.add_argument(
        "--mask_method", help="The classification thresholds to use."
                              "Can only be one of 'L1', 'L2', 'both'.",
        default='L1', type=str)
    parser.add_argument(
        "--tile", help="Size of tile.",
        default=32, type=int)
    parser.add_argument(
        "--channel", help="Channel of image.",
        default=0, type=int)
    parser.add_argument(
        "--samples", help="Number of random tile samples.",
        default=50, type=int)
    parser.add_argument(
        "--ref_rate", help="Reference down sampling rate.",
        default=2, type=int)
    parser.add_argument(
        "--ref_method", help="The reference interpolation to use."
                              "Can only be one of 'AR', 'NN', 'BL', 'BC'.",
        default='AR', type=str)
    parser.add_argument(
        "--threshold", help="Threshold on the relative DCT value. "
                            "Effective only when div_dct is True.",
        default=20, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    classify_folder(args.input_folder, args.output_filename, args.tile, args.channel, args.samples,
                    args.ref_rate, args.ref_method, args.threshold, size=args.max_image, method=args.mask_method)
