import numpy as np
import cv2


x_choices = np.linspace(0, 1, 32)
y_choices = np.linspace(0, 1, 32)
scale_choices = np.linspace(0.5, 1, 6)
rot_choices = np.linspace(0, 360, 40)

cov = [[1, .5, 0, 0],
       [.5, 1, 0, 0],
       [0, 0, 1, .5],
       [0, 0, .5, 1]]


def resize_images(X):
    X_resized = np.zeros([X.shape[0], 64, 64, X.shape[-1]], dtype=X.dtype)
    for i, x in enumerate(X):
        X_resized[i] = cv2.resize(x, (64, 64), interpolation=cv2.INTER_AREA)[:, :, None]
    return X_resized


def draw_ellipse(x, y, scale, rot):
    x = x_choices[np.digitize(x, x_choices) - 1]
    y = y_choices[np.digitize(y, y_choices) - 1]
    center = (int(round(x * 156 + 50)), int(round(y * 156 + 50)))
    img = np.zeros((256, 256, 1), np.uint8)
    scale = scale / 2. + .5
    scale = scale_choices[np.digitize(scale, scale_choices) - 1]
    size = (int(round(scale * 50)), int(round(scale * 25)))
    rot *= 360
    angle = rot_choices[np.digitize(rot, rot_choices) - 1]
    cv2.ellipse(img, center, size, angle, 0, 360, [255] * 3, -1, )
    return img


def latent2image(latents):
    size = len(latents)
    images = []
    for i in range(size):
        im = draw_ellipse(*latents[i])
        images.append(im)
    images = np.array(images)
    return resize_images(images)


def sample(size=100, normalize=False):
    mean = np.zeros(4)
    z = np.random.multivariate_normal(mean, cov, size)
    z = 0.5 * (np.clip(z, -1, 1) + 1)
    images = latent2image(z)
    if normalize:
        images = images / 255.
    return images


def conditional_sample(z_idx, size=100, normalize=False):
    assert (z_idx < 4)
    mean = np.zeros(4)
    z = np.random.multivariate_normal(mean, cov, size)
    z = 0.5 * (np.clip(z, -1, 1) + 1)
    z[:, z_idx] = z[0, z_idx]
    images = latent2image(z)
    if normalize:
        images = images / 255.
    return images.reshape(size, 1, 64, 64)
