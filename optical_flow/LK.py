import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def lucas_kanade_step(im1: np.ndarray, im2: np.ndarray, window_size):
    if im1.dtype == 'uint8':
        im1 = im1.astype(float) / 255
    if im2.dtype == 'uint8':
        im2 = im2.astype(float) / 255
    # Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3)
    # Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3)
    Iy, Ix = np.gradient(im2)
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy
    It = im2 - im1
    IxIt = Ix * It
    IyIt = Iy * It

    IxIx_sum = convolve2d(IxIx, np.ones((window_size, window_size)), mode='same')
    IyIy_sum = convolve2d(IyIy, np.ones((window_size, window_size)), mode='same')
    IxIy_sum = convolve2d(IxIy, np.ones((window_size, window_size)), mode='same')
    IxIt_sum = convolve2d(IxIt, np.ones((window_size, window_size)), mode='same')
    IyIt_sum = convolve2d(IyIt, np.ones((window_size, window_size)), mode='same')

    determinate = IxIx_sum * IyIy_sum - IxIy_sum ** 2
    u = (IxIy_sum * IyIt_sum - IyIy_sum * IxIt_sum) / determinate
    v = (IxIy_sum * IxIt_sum - IxIx_sum * IyIt_sum) / determinate

    u = np.nan_to_num(u)
    v = np.nan_to_num(v)
    return u, v


def wrap_image(image, u, v):
    rows, cols = image.shape
    x_mesh, y_mesh = np.meshgrid(np.arange(cols), np.arange(rows))
    x_trans = x_mesh + u
    y_trans = y_mesh + v

    # image_wraped = griddata((x_trans.flatten(), y_trans.flatten()), image.flatten(), (x_mesh.flatten(), y_mesh.flatten()))
    image_wraped = griddata((y_mesh.flatten(), x_mesh.flatten()), image.flatten(),
                            (y_trans.flatten(), x_trans.flatten()))
    image_wraped = np.reshape(image_wraped, image.shape)
    image_wraped[np.isnan(image_wraped)] = image[np.isnan(image_wraped)]
    return image_wraped


def lucas_kanade_pyramid(im1, im2, window_size, max_iteration, num_of_levels):
    # get pyramids
    im1_pyramid = []
    im2_pyramid = []
    im1_pyramid.append(im1)
    im2_pyramid.append(im2)
    distance = np.inf
    for ii in range(1, num_of_levels):
        im1_pyramid.append(cv2.pyrDown(im1_pyramid[-1]))
        im2_pyramid.append(cv2.pyrDown(im2_pyramid[-1]))
    for level in range(num_of_levels-1, -1, -1):
        rows, cols = im2_pyramid[level].shape
        if level == num_of_levels - 1:
            u = np.zeros((rows, cols))
            v = np.zeros((rows, cols))
        else:
            u = cv2.resize(2 * u, (cols, rows))
            v = cv2.resize(2 * v, (cols, rows))
        iteration_number = 0
        while iteration_number <= max_iteration:
            im2_wrap = wrap_image(im2_pyramid[level], u, v)
            du, dv = lucas_kanade_step(im1_pyramid[level], im2_wrap, window_size)
            new_distance = np.sum(du ** 2 + dv ** 2)
            if new_distance < distance:
                u = u + du
                v = v + dv
                distance = new_distance
                iteration_number += 1
            else:
                break
    return u, v


if __name__ == '__main__':
    images = np.load('./data/images.npz')
    im1 = images['I1']
    im2 = images['I2']
    # u, v = lucas_kanade_step(im1, im2, 5)
    u, v = lucas_kanade_pyramid(im1, im2, 5, 3, 3)
    im1_from_im2 = wrap_image(im2, u, v)
    plt.figure()
    plt.imshow(im1_from_im2, cmap='gray')
    plt.title('wrap')
    plt.figure()
    plt.imshow(im1, cmap='gray')
    plt.title('image 1')
    plt.figure()
    plt.imshow(im2, cmap='gray')
    plt.title('image 2')