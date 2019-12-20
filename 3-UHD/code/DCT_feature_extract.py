import argparse
import os
from DCT_Judger import extract_feature_folder


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
    parser.add_argument(
        "--offset", help="Shift the relative DCT value to left by 'offset'.",
        default=0, type=float)
    parser.add_argument(
        "--max_image", help="Divide reference before statistics.",
        default=None, type=int)
    parser.add_argument(
        "--ref_method", help="Interpolation in reference images.",
        default='AR', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    feats = extract_feature_folder(args.folder, args.tile, args.channel, args.samples,
                                   args.ref_rate, args.threshold, args.div_ref,
								   offset=args.offset, size=args.max_image, ref_method=args.ref_method)
    np.savetxt(os.path.join(args.folder, "tile_{}_channel_{}_samples_{}_rate_{}_threshold_{}_div_{}"
                                         "_offset_{}_ref_{}.txt".format(args.tile, args.channel, args.samples,
                                                                        args.ref_rate, args.threshold, args.div_ref,
                                                                        args.offset, args.ref_method)), feats)
