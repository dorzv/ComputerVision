""" use template match with normalized correlation to find Woldo"""

import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import cv2

def norm_cross_correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    return the normalized cross correlation between gray image and template
    Args:
        image (np.ndarray): the image in which we search for the template match
        template (np.ndarray): the template to be found in the image

    Returns:
        (np.ndarray): normalized cross correlation matrix between the image and the template.
            The maximum of the matrix should be the center of the match of the template.
    """
    temp_width = template.shape[1]
    temp_height = template.shape[0]

    # if the template shape is not of odd number, cut it
    if temp_width % 2 == 0:
        template = template[:, :-1]
        temp_width -= 1
    if temp_height % 2 == 0:
        template = template[:-1, :]
        temp_height -= 1

    zero_mean_template = template - np.mean(template)

    # zero padding the image
    image_pad = np.pad(image, (2 * (temp_height//2, ), 2 * (temp_width // 2, )))
    normalized_cross_correlation = np.zeros_like(image)

    for u, x in enumerate(range(temp_width // 2, image_pad.shape[1] - temp_width // 2)):
        for v, y in enumerate(range(temp_height // 2, image_pad.shape[0] - temp_height // 2)):
            roi = image_pad[y-temp_height//2:y+temp_height//2+1, x-temp_width//2:x+temp_width//2+1]
            zero_mean_roi  = roi - np.mean(roi)

            normalized_cross_correlation[v, u] = np.sum(zero_mean_roi * zero_mean_template) / np.sqrt(np.sum(zero_mean_roi ** 2) * np.sum(zero_mean_template ** 2))


    return normalized_cross_correlation



# load image and template as float image between 0 and 1
im = cv2.imread('Wheres-waldo.jpg').astype(np.float32) / 255
template = cv2.imread('waldo.jpg').astype(np.float32) / 255

# convert BGR to RGB
im = im[..., ::-1]
template = template[..., ::-1]

im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

cross_correlation = norm_cross_correlation(im_gray, template_gray)
max_correlation = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
marked_image = im.copy()

cv2.rectangle(marked_image,
              (max(max_correlation[1] - template.shape[1] // 2, 0), max(max_correlation[0] - template.shape[0] // 2, 0)),
               (min(max_correlation[1] + template.shape[1] // 2, im.shape[1] - 1), min(max_correlation[0] + template.shape[0] // 2, im.shape[0] - 1)),
               color=(1, 0, 0), thickness=2)

plt.imshow(marked_image)