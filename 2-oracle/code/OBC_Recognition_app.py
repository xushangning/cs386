import argparse


def run(folder, output_fname, method, model_path, num_cat, tm_method):
    """ Main function that runs the classifier.
    """
    assert method in ['DNN', 'TM'], "Method can only be 'DNN' or 'TM'."
    assert tm_method in ['MSE', 'MSE_NORMED', 'CCORR', 'CCORR_NORMED'],\
        "Method for template matching can only be one of 'MSE', 'MSE_NORMED', 'CCORR', 'CCORR_NORMED'"
    format = output_fname.split('.')[-1]
    assert format in ['csv', 'txt'], "The format of the output file should be either 'csv' or 'txt'."

    import numpy as np
    import pandas as pd
    from Dataset import Dataset
    from keras.models import load_model
    from template_matching import TemplateMatch

    # load data
    if method == 'DNN':
        data, fnames = Dataset.get_image_folder(folder, names=True, normalize=True)
        data[data < 0.5] = 0
    elif method == 'TM':
        # fix model for template matching
        model_path = "model/templ.pkl"
        data, fnames = Dataset.get_image_folder(folder, names=True, normalize=False)

    # generate prediction
    if method == 'DNN':
        model = load_model(model_path)
        prob = model.predict(data)
        y = np.argmax(prob, axis=1)
    elif method == 'TM':
        tm = TemplateMatch()
        tm.load_model(model_path)
        y = tm.predict(data, num_cat, tm_method)

    # save prediction to file
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
        "--model_path", help="Path to model for classification. Model should match 'method' and 'num_cat'."
                             "Auto adjustment for TM method.",
        default='model/weights_norm_cat10_v2.hdf5', type=str)
    parser.add_argument(
        "--num_cat", help="Number of categories to classify. Either 10 or 40.",
        default=10, type=int)
    parser.add_argument(
        "--tm_method", help="Method for template matching to calculate the score."
                            "Should be one of the 'MSE', 'MSE_NORMED', 'CCORR', 'CCORR_NORMED'.",
        default='MSE', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.input_folder, args.output_filename, args.method,
        args.model_path, args.num_cat, args.tm_method)
