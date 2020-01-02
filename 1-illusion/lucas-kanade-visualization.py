# no need to scale the image, because LaTeX will do it for you
import cv2 as cv

FLOW_SCALE_FACTOR = 60

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
red = (0, 0, 255)
circle_color = (0, 255, 0)


def visualize(group: str):
    template_path = 'dataset/' + group + '/img{}-{}.png'

    for i in range(2):
        old_frame = cv.imread(template_path.format(i, 0))
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        for j in range(7, 8):
            drawings = old_frame.copy()
            frame = cv.imread(template_path.format(i, j))
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            for new, old in zip(good_new, good_old):
                offset = (old - new) * FLOW_SCALE_FACTOR
                old = tuple(new + offset)
                new = tuple(new)
                cv.line(drawings, new, old, red, 2)
                cv.circle(drawings, new, 2, circle_color, -1)
            cv.imwrite('lk-' + group + '-{}.png'.format(i), drawings)


if __name__ == '__main__':
    visualize('control')
    visualize('rotate')
