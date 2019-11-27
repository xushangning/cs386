import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset


def get_template(group):
    return group.mean(axis=0)


# mse error
def score(group, template, method):
    assert method in ['MSE', 'SQDIFF', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED']

    # custom implement
    if method == 'MSE':
        return np.array([np.mean((group[i] - template) ** 2) for i in range(group.shape[0])])
        # return np.mean((group - template) ** 2, axis=(1, 2))

    # use cv2.matchTemplate
    else:
        # move template around to get the best match
        res = [cv2.matchTemplate(group[i], template.astype(np.uint8), eval('cv2.TM_' + method)) for i in
               range(group.shape[0])]
        minmax = np.array([cv2.minMaxLoc(res[i]) for i in range(group.shape[0])])
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
        g, _ = Dataset.get_image_folder('0' + str(102 + i), 40)
        groups.append(g)
        temp = get_template(g)
        templates.append(temp)
        # plt.imshow(temp,cmap='gray')
        # plt.show()

    for method in methods:
        print(method)

        hits40 = []
        hits10 = []
        total = []
        for i in range(40):
            m = match(groups[i], templates, method)
            hits40.append(np.sum(m == i))
            hits10.append(np.sum(m // 4 == i // 4))
            total.append(len(m))
            # print(i, np.mean(m == i))
            break

        print('cat40', 1.0 * sum(hits40) / sum(total))
        print('cat10', 1.0 * sum(hits10) / sum(total))


if __name__ == '__main__':
    main()
    # g, _ = Dataset.get_image_folder('0' + str(102), 40)
    # # groups.append(g)
    # temp = get_template(g)
    # s = score(g, temp, 'MSE')
    # print(s)
