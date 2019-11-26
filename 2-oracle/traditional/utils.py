import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# label: [0,39]
def get_group(label, size=(93, 81)):
    DIR = "data/0" + str(102 + label)
    group = np.zeros((len(os.listdir(DIR)), *size))
    for i, filename in enumerate(os.listdir(DIR)):
        path = os.path.join(DIR, filename)
        img = cv2.imread(path, 0)
        img.resize(size)
        group[i] = img
    return group


def get_template(group):
    return group.mean(axis=0)


# mse error
def score(group, template):
    return np.array([1 / np.mean((group[i] - template) ** 2) for i in range(group.shape[0])])


def match(dataset, templates):
    scores = np.array([score(dataset, t) for t in templates])
    return scores.argmax(axis=0)


if __name__ == '__main__':
    g = get_group(0)
    t = get_template(g)
    print(score(g,t).shape)
    print(match(g,[t,t]).shape)