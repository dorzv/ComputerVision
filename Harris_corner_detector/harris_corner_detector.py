import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import cv2


def corner_detector(image, window_size, k, threshold):
    image_copy = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

    Ix = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    offset = window_size // 2
    image_R_matrix = np.zeros((height, width))
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            M_00 = np.sum(IxIx[y-offset:y+offset+1, x-offset:x+offset+1])
            M_11 = np.sum(IyIy[y-offset:y+offset+1, x-offset:x+offset+1])
            M_01 = np.sum(IxIy[y-offset:y+offset+1, x-offset:x+offset+1])
            H = np.array([[M_00, M_01], [M_01, M_11]])
            R = det(H) - k * (np.trace(H) ** 2)
            image_R_matrix[y, x] = R

    cv2.normalize(image_R_matrix, image_R_matrix, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    is_corner = image_R_matrix > threshold

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if is_corner[y, x]:
                cv2.circle(image_copy, (x, y), 1, (0, 0, 255), -1)

    return image_copy


image = cv2.imread('./im.jpg')
corner_image = corner_detector(image, 5, 0.04, 0.25)
plt.imshow(corner_image[:, :, ::-1])
plt.show()
print('dor')
