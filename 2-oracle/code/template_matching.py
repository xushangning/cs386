import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset


def get_template(group):
    return group.mean(axis=0)


# mse error
def score(group, template):
    return np.array([1 / np.mean((group[i] - template) ** 2) for i in range(group.shape[0])])


def match(dataset, templates):
    scores = np.array([score(dataset, t) for t in templates])
    return scores.argmax(axis=0)


templates = []
groups = []
for i in range(40):
    g, _ = Dataset.get_image_folder('0' + str(102 + i), 40)
    groups.append(g)
    temp = get_template(g)
    templates.append(temp)
    # plt.imshow(temp,cmap='gray')
    # plt.show()

for i in range(40):
    m = match(groups[i], templates)
    print(i, np.mean(m == i))
