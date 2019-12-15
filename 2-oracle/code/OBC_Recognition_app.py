import argparse
import numpy as np
import pandas as pd
from Dataset import Dataset
from keras.models import load_model
from template_matching import TemplateMatch


def run(folder, output_fname, method, model_path, num_cat):
    assert method in ['DNN', 'TM'], "Method can only be 'DNN' or 'TM'."
    format = output_fname.split('.')[-1]
    assert format in ['csv', 'txt'], "The format of the output file should be either 'csv' or 'txt'."

    if method == 'DNN':
        data, fnames = Dataset.get_image_folder(folder, num_cat, names=True, normalize=True)
        data[data < 0.5] = 0
    elif method == 'TM':
        data, fnames = Dataset.get_image_folder(folder, num_cat, names=True, normalize=False)

    if method == 'DNN':
        model = load_model(model_path)
        prob = model.predict(data)
        y = np.argmax(prob, axis=1)
    elif method == 'TM':
        tm = TemplateMatch()
        tm.load_model(model_path)
        y = tm.match(data)

    if format == 'csv':
        df = pd.DataFrame({'fname': fnames, 'label': y})
        df.to_csv(output_fname, index=False)
    elif format == 'txt':
        np.savetxt(output_fname, y)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="OBC Recognition")
    parser.add_argument("--input_folder", help="Input image folder.", type=str)
    parser.add_argument(
        "--output_filename", help="The output txt file of classification results.",
        default="output.csv", type=str)
    parser.add_argument(
        "--method", help="Method for classification. Either 'DNN' or 'TM'.",
        default='DNN', type=str)
    parser.add_argument(
        "--model_path", help="Path to model for classification. Model should match 'method' and 'num_cat'.",
        default='model/weights_norm_cat10_v2.hdf5', type=str)
    parser.add_argument(
        "--num_cat", help="Number of categories to classify. Either 10 or 40",
        default=10, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.input_folder, args.output_filename, args.method,
        args.model_path, args.num_cat)
