import os
import argparse



def complete_feature_extract(img, tile, channel=0, samples=50, threshold=20):
    if len(img.shape) > 3:
        img = img[:, :, channel]

    xs = list(range(img.shape[0] // tile))
    ys = list(range(img.shape[1] // tile))
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    idxs = list(zip(xs, ys))[:samples]

    abs_dev = lambda x: abs(x).mean()
    sqr_dev = lambda x: (x ** 2).mean() ** 0.5

    # compute raw DCT
    raw_dcts = []
    abd = 0
    std = 0
    for i, j in idxs:
        dct = dct_tile(img[:, :], i, j, tile, ref_int=None)
        raw_dcts.append(dct)
        abd += abs_dev(dct[dct < threshold])
        std += sqr_dev(dct[dct < threshold])
    raw_dcts = np.array(raw_dcts)
    raw_feat = [abd / len(idxs), std / len(idxs)]

    # compute relative DCT with rate 2
    ref_method_dict = {
        'AR': cv2.INTER_AREA,
        'NN': cv2.INTER_NEAREST,
        'BL': cv2.INTER_LINEAR,
        'BC': cv2.INTER_CUBIC,
    }
    feature = []
    for ref in ['AR', 'NN', 'BL', 'BC']:
        abds = []
        stds = []
        for k, (i, j) in enumerate(idxs):
            dct = dct_tile(img[:, :], i, j, tile, rate=2, origin=raw_dcts[k, :, :],
                           ref_int=ref_method_dict[ref]).flatten() - 1
            dct = dct[abs(dct) < threshold]
            abd = abs_dev(dct)
            std = sqr_dev(dct)
            abds.append(abd)
            stds.append(std)
        feature += [np.nanmean(abds), np.nanmean(stds)]

    # append raw DCT feature
    feature += raw_feat

    # compute DCT with rate 2 method AR
    abds = []
    stds = []
    for k, (i, j) in enumerate(idxs):
        dct = dct_tile(img[:, :], i, j, tile, rate=3, origin=raw_dcts[k, :, :],
                       ref_int=ref_method_dict['AR']).flatten() - 1
        dct = dct[abs(dct) < threshold]
        abd = abs_dev(dct)
        std = sqr_dev(dct)
        abds.append(abd)
        stds.append(std)
    feature += [np.nanmean(abds), np.nanmean(stds)]

    return feature


def extract_feature_folder(folder, tile, channel=0, samples=50, threshold=20,
                           return_fnames=False, size=None):
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
        feat = complete_feature_extract(img, tile, channel, samples, threshold)
        feats.append(feat)

    if return_fnames:
        return np.array(feats), fnames
    return np.array(feats)


def classify_folder(clf, folder, output_file, tile, channel=0, samples=50, threshold=20, size=None):
    feats, fnames = extract_feature_folder(folder, tile, channel, samples, threshold,
                                           return_fnames=True, size=size)
    res = clf.predict(feats)
    if 'csv' in output_file:
        pd.DataFrame({'fname': fnames, 'label': res}).to_csv(output_file, index=False)
    else:
        np.savetxt(output_file, res)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="SVM Classifier")
    parser.add_argument("--input_folder", help="Input image folder.", type=str)
    parser.add_argument(
        "--output_filename", help="The output txt file of classification results.",
        default="output.txt", type=str
    )
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
        "--threshold", help="Threshold on the relative DCT value. "
                            "Effective only when div_dct is True.",
        default=20, type=int)
    parser.add_argument(
        "--max_image", help="Divide reference before statistics.",
        default=None, type=int)
    parser.add_argument(
        "--model_path", help="The path to SVM model.",
        default="./svm_complete.model", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    import cv2
    import numpy as np
    import pandas as pd
    from DCT_Judger import dct_tile, get_image
    from sklearn.externals import joblib

    clf = joblib.load(args.model_path)
    classify_folder(clf, args.input_folder, args.output_filename,
                    args.tile, args.channel, args.samples, args.threshold, args.max_image)

