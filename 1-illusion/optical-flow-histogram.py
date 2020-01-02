import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


def visualize(group_name, n_img):
    template_path = '/home/shining/school/cs386/results/pred_' + group_name + '/img{}-{}.png'

    bin_edges = np.linspace(-2, 2, 11)
    hist = np.zeros(10)

    for i in range(n_img):
        old_gray = cv.imread(template_path.format(i, 0), cv.IMREAD_GRAYSCALE)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        for j in range(7, 8):
            frame_gray = cv.imread(template_path.format(i, j), cv.IMREAD_GRAYSCALE)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            log_magnitudes = tuple(
                np.log10(np.square(new - old).sum()) / 2
                for new, old in zip(good_new, good_old)
            )
            temp_hist, _ = np.histogram(log_magnitudes, bin_edges)
            hist += temp_hist
    hist /= n_img

    plt.ylim(top=31)
    plt.hist(bin_edges[:-1], bin_edges, weights=hist)
    plt.xlabel('Lg of magnitudes')
    plt.title(group_name)


if __name__ == '__main__':
    visualize('rotate', 17)
    plt.savefig('flow-mag-plot-rotate.pdf')
    plt.close()
    visualize('control', 13)
    plt.savefig('flow-mag-plot-control.pdf')
