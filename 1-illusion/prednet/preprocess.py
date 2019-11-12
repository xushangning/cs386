import numpy as np
from PIL import Image, ImageSequence
import os


# get gif images from GIF_DIR
def get_test_data(GIF_DIR, nt=10, size=(160, 128)):
    test_data = []
    for filename in os.listdir(GIF_DIR):
        im = Image.open(os.path.join(GIF_DIR, filename))

        frames = []
        index = 0
        for frame in ImageSequence.Iterator(im):
            # print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
            if index >= nt:
                break
            frames.append(frame.resize(size, Image.ANTIALIAS))
            index += 1
            # frame.save("./imgs/frame%d.png" % index)

        seq = [np.asarray(frame.convert('RGB')) for frame in frames]

        if index >= nt:
            # print(np.array(seq).shape)
            test_data.append(seq)

    test_data = np.array(test_data)
    # print(test_data.shape)
    # test_data = test_data.transpose((0, 1, 4, 2, 3))
    test_data = test_data / 255
    print(test_data.shape)
    # print(test_data)
    return test_data


# get static pictures from IMG_DIR
def get_static_test_data(IMG_DIR, nt=10, size=(160, 128)):
    test_data = []
    for filename in os.listdir(IMG_DIR):
        im = Image.open(os.path.join(IMG_DIR, filename))

        frames = [im.resize(size, Image.ANTIALIAS)] * nt
        # frame.save("./imgs/frame%d.png" % index)

        seq = [np.asarray(frame.convert('RGB')) for frame in frames]

        # print(np.array(seq).shape)
        test_data.append(seq)

    test_data = np.array(test_data)
    # print(test_data.shape)
    # test_data = test_data.transpose((0, 1, 4, 2, 3))
    test_data = test_data / 255
    print(test_data.shape)
    # print(test_data)
    return test_data

# get_static_test_data()
