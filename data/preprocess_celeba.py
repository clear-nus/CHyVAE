import numpy as np
import cv2
import glob
from tqdm import tqdm


def preprocess_split(name, size=64.):
    images = glob.glob('./tmp/CelebA/splits/%s/*.jpg' % name)
    in_height, inwidth = cv2.imread(images[0]).shape[:-1]
    out_width = int(size)
    out_height = int(in_height * size / inwidth)
    from_y = int((out_height - out_width) / 2)
    upto_y = from_y + out_width
    images_arr = np.zeros([len(images), out_width, out_width, 3])
    for i, name in tqdm(enumerate(images)):
        img = cv2.resize(cv2.imread(name), (out_width, out_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[from_y:upto_y, :]
        images_arr[i] = img
    return images_arr


if __name__ == '__main__':
    train_images = preprocess_split('train', size=64.).astype(np.uint8)
    test_images = preprocess_split('test', size=64.).astype(np.uint8)
    np.save('celeba/train_X.npy', train_images)
    np.save('celeba/test_X.npy', test_images)
