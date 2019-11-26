import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


templates = []
groups = []
for i in range(40):
    g = get_group(i)
    groups.append(g)
    temp = get_template(g)
    templates.append(temp)
    # plt.imshow(temp,cmap='gray')
    # plt.show()

for i in range(40):
    m = match(groups[i], templates)
    print(i, np.mean(m == i))
