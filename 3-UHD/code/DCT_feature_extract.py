import argparse
import os
from DCT_Judger import *


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="DCT Feature")
    parser.add_argument("folder", help="Image folder.", type=str)
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
        "--threshold", help="Threshold on the relative DCT value. "
                            "Effective only when div_dct is True.",
        default=20, type=int)
    parser.add_argument(
        "--div_ref", help="Divide reference before statistics.",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    feats = extract_feature_folder(args.folder, args.tile, args.channel, args.samples,
                                   args.ref_rate, args.threshold, args.div_ref)
    np.savetxt(os.path.join(args.folder, "tile_{}_channel_{}_samples_{}_rate_{}_threshold_{}_div_{}.txt"
               .format(args.tile, args.channel, args.samples,
                       args.ref_rate, args.threshold,
                       args.div_ref)), feats)
