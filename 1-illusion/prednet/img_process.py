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


def get_raw_images(IMG_DIR, size=(160, 128)):
    raw = []
    for filename in os.listdir(IMG_DIR):
        im = Image.open(os.path.join(IMG_DIR, filename))

        im = im.resize(size, Image.ANTIALIAS)
        # frame.save("./imgs/frame%d.png" % index)

        raw.append(np.asarray(im.convert('RGB')))

    raw = np.array(raw)
    raw = raw / 255
    print(raw.shape)
    return raw


def save_images_from_np(X_hat, raw_img, outname):
    if not os.path.exists('results/pred_'+outname):
        os.mkdir('results/pred_'+outname)
    for j in range(X_hat.shape[0]):
        print('saving image ' + str(j) + '...')
        im = Image.fromarray((raw_img[j]*255).astype(np.uint8))
        im.save('results/pred_'+outname+'/img' + str(j) + '-0.png')

        frames = X_hat[j]
        for i in range(1, frames.shape[0]):
            im = Image.fromarray((frames[i]*255).astype(np.uint8))
            im.save('results/pred_'+outname+'/img' + str(j) + '-' + str(i) + '.png')
            # plt.imshow(frames[i])
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig('results/output/img' + str(j) + '-' + str(i) + '.png', bbox_inches='tight', pad_inches=0.0)
            # plt.show()

def main():
    X_hat = np.load('./results/pred_rotate0.npy')
    raw_img = get_raw_images('test_data/static')
    save_images_from_np(X_hat, raw_img, 'rotate0')


if __name__ == '__main__':
    main()
