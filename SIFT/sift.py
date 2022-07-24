# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
# https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

from functools import cmp_to_key
import numpy as np
import cv2

float_tolerance = 1e-7


def generate_base_pyramid_image(image, final_sigma, sigma_1=0.5):
    """
    Generate the base image for the pyramid. This image is twice
    the size of the original image. Also we assume the original
    image has a blurr of σ_1 = 0.5 (the minimum needed to prevent
    significant aliasing)
    Args:
        image (np.ndarry): The original image
        final_sigma (float): The sigma of the base image in the pyramid.
            final_sigma ** 2 = sigma_1 ** 2 + sigma_2 ** 2
        sigma_1 (float): The sigma the original image assumed to have

    Returns:
        (np.ndaaray): the base image of the pyramid, which is twice the
            size of the original image, and has sigma of final sigma
    """

    image = cv2.resize(image, None, fx=2, fy=2)
    sigma_2 = np.sqrt(final_sigma ** 2 - (2 * sigma_1) ** 2)
    final_blurred_im = cv2.GaussianBlur(image, (0, 0), sigma_2)
    return final_blurred_im


def get_number_of_pyramid_levels(image):
    """
    Get the number of levels in the pyramid
    Args:
        image (np.ndarray): the original image

    Returns:
        (int): the number of levels in the pyramid
    """
    return int(np.round(np.log(np.min(image.shape)) / np.log(2) - 1))


def get_octave_gaussian_sigma(sigma, s):
    """
    According to section 3, we need to build in each octave a series of
    blurred image such that σ_{i+1} = k * σ_{i}. This function returns
    the σ of each Gaussian need to be applied on the previous image to
    get a blurred image with the correct σ.
    Args:
        sigma (float): the σ of the first image blurring
        s (int): number of interval of blurred images in an octave

    Returns:
        (np.ndarray): an array with the σ's of each Gaussian filter
            need to be applied on the images in the octave
    """
    images_in_octave = s + 3
    k = 2 ** (1 / s)  # ecah blur of iamge in the octave has σ = k * σ_pre, where σ_pre is the σ of the gaussian blur of the previous image
    gaussian_sigmas = np.zeros(images_in_octave)  # the sigma of each Gaussian blurring need to be applied to the previuos image to get the new blurred image in the octave (which is kσ_pre)
    gaussian_sigmas[0] = sigma
    k_multiplication = k ** np.arange(0, images_in_octave)
    sigma_total = sigma * k_multiplication
    gaussian_sigmas[1:] = np.sqrt(sigma_total[1:] ** 2 - sigma_total[:-1] ** 2)
    return gaussian_sigmas


def generate_scale_space_pyramid(image, num_of_octave, gaussian_sigmas):
    """
    Create the scale-space pyramid. In each level of the pyramid the
    base image is blurred repeatedly according to the sigmas of the
    Gaussian filter we calculated earlier. After each level calculation
    the image which has twice the sigma of the base image is resampled
    by half, and we repeat the process for the next level.
    Args:
        image (np.ndarray): the base image
        num_of_octave (int): number of levels of the pyramid
        gaussian_sigmas (np.array[float]): the sigmas need to be applied
            on each image in each level

    Returns:
        (list): the pyramid structure of blurred images
    """
    pyramid = []
    for octave_ii in range(num_of_octave):
        octave_images = [image]
        for sigma in gaussian_sigmas[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
            octave_images.append(image)
        pyramid.append(octave_images)
        image_to_resample = octave_images[-3]  # resample the Gaussian image that has twice the initial value of σ
        image = cv2.resize(image_to_resample, (int(image_to_resample.shape[1] / 2), int(image_to_resample.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    return pyramid


def generate_dog_pyramid(pyramid):
    """
    Create Difference of Gaussian (DoG) pyramid. Each two adjacent images
    in each level are subtracted from each other to get the DoG of them.
    Args:
        pyramid (list): The pyramid structure of the Gaussian blurred images

    Returns:
        (list): a pyramid structure of DoG
    """
    dog_pyramid = [[cv2.subtract(im2, im1) for im1, im2 in zip(octave_images[:-1], octave_images[1:])]
                   for octave_images in pyramid]
    return dog_pyramid


def is_pixel_extrema(im1, im2, im3, threshold):
    """
    Check if a pixel is an extrema. The canter pixel of the middle image
    (im2) is compared to the other pixels, and return true if it's a maxima
    or a minima.
    Args:
        im1 (np.ndarray): part of the image before im2 in the same level of
            the pyramid (less blurred) with 3x3 size.
        im2 (np.ndarray): the part of the image of size 3x3 with the
            candidate pixel in the middle.
        im3 (np.ndarray): part of the image after im2 in the same level of
            the pyramid (more blurred) with 3x3 size.
        threshold (float): the threshold a candidate need to be above
            (in absolute value) to be considered as extrema

    Returns:
        (bool): True if the candidate pixel is an extrema
    """
    candidate_pixel = im2[1, 1]
    if np.abs(candidate_pixel) > threshold:
        return candidate_pixel == np.max([im1, im2, im3]) if candidate_pixel > 0 \
            else candidate_pixel == np.min([im1, im2, im3])
    return False


def scale_space_extrema_detection(dog_pyramid, blurred_images_pyramid, num_intervals, dist_from_boarder, sigma,
                                  contrast_threshold=0.04):
    """
    find keypoints in the scale space of the image, localized them using quadratic
    fitting and add orientation.
    Args:
        dog_pyramid (list): a pyramid structure of DoG
        blurred_images_pyramid (list): a pyramid structure of blurred images
        num_intervals (int): number of interval of blurred images in an octave
        dist_from_boarder (int): The number of pixels to keep from the
            image's boarder when searching for extrema
        sigma (float): the sigma of the blurred of the base image.
        contrast_threshold:

    Returns:
        (list[cv2.KeyPoints]): a list of keypoints
    """
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_ii, dog_in_octave in enumerate(dog_pyramid):
        for image_ii, (im1, im2, im3) in enumerate(zip(dog_in_octave, dog_in_octave[1:], dog_in_octave[2:]), start=1):
            for y in range(dist_from_boarder, im1.shape[0] - dist_from_boarder):
                for x in range(dist_from_boarder, im1.shape[1] - dist_from_boarder):
                    if is_pixel_extrema(im1[y-1:y+2, x-1:x+2], im2[y-1:y+2, x-1:x+2], im3[y-1:y+2, x-1:x+2], threshold):
                        extrema_localization = localize_by_quadratic_fit(y, x, image_ii, octave_ii, dist_from_boarder,
                                                                         num_intervals, dog_in_octave, sigma)
                        if extrema_localization:
                            keypoint, localized_image_index = extrema_localization
                            keyposints_with_orantaions = add_orientation_to_key_point(keypoint, octave_ii,
                                                          blurred_images_pyramid[octave_ii][localized_image_index])
                            for keypoint_with_orientation in keyposints_with_orantaions:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def gradient_at_center_pixel(pixels_cube):
    """
    Get the gradient at the central pixel of a 3x3x3 pixels cube. The
    derivative calculated as (f(x+h) - f(x-h)) / (2h). The first axis
    is the scale, the second is y and the third is x
    Args:
        pixels_cube (nd.array): a 3x3x3 pixels of the pixels around the
            candidate pixel.

    Returns:
        (np.ndarray): array contains the derivative is the scale, y and x
            directions
    """
    dx = 0.5 * (pixels_cube[1, 1, 2] - pixels_cube[1, 1, 0])
    dy = 0.5 * (pixels_cube[1, 2, 1] - pixels_cube[1, 0, 1])
    ds = 0.5 * (pixels_cube[2, 1, 1] - pixels_cube[0, 1, 1])
    return np.array([dx, dy, ds])


def hessian_at_center_pixel(pixels_cube):
    """
    Get the hessian matrix at the central pixel of a 3x3x3 pixels cube.
    The second derivative calculated as f''(x) = (f(x+h) - 2f(x) + f(x-h)) / (h ** 2),
    and d^2 f(x,y) / dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    The first axis is the scale, the second is y and the third is x
    Args:
        pixels_cube (nd.array): a 3x3x3 pixels of the pixels around the
            candidate pixel.

    Returns:
        (np.ndarray): the hessian matrix
    """

    center_pixel_value = pixels_cube[1, 1, 1]
    dxx = pixels_cube[1, 1, 2] - 2 * center_pixel_value + pixels_cube[1, 1, 0]
    dyy = pixels_cube[1, 2, 1] - 2 * center_pixel_value + pixels_cube[1, 0, 1]
    dss = pixels_cube[2, 1, 1] - 2 * center_pixel_value + pixels_cube[0, 1, 1]
    dxy = 0.25 * (pixels_cube[1, 2, 2] - pixels_cube[1, 2, 0] - pixels_cube[1, 0, 2] + pixels_cube[1, 0, 0])
    dxs = 0.25 * (pixels_cube[2, 1, 2] - pixels_cube[2, 1, 0] - pixels_cube[0, 1, 2] + pixels_cube[0, 1, 0])
    dys = 0.25 * (pixels_cube[2, 2, 1] - pixels_cube[2, 0, 1] - pixels_cube[0, 2, 1] + pixels_cube[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def localize_by_quadratic_fit(y, x, image_index, octave_index, dist_from_boarder, num_intervals, dog_images_in_octave, sigma,  max_iteration=5, eigenvalue_ratio=10):
    """
    Better localize the extrema point in the spatial-scale space using
    quadratic fit as describe in section 4 of the paper.
    Args:
        y (int): y position of the extrema candidate.
        x (int): x position of the extrema candidate.
        image_index (int): the image index in the octave of the extrema
            candidate.
        octave_index (int): the octave index of the extrema candidate.
        dist_from_boarder (int): the distance in pixels to keep from
            the image boarders.
        num_intervals (int): the number of intervals (s) in each octave.
        dog_images_in_octave (list): the Differnce of Gaussian images in
            the octave.
        sigma (float): the sigma of the blurred of the base image.
        max_iteration (int): the maximum iteration to try until converge
        eigenvalue_ratio (float): the ratio of the eigenvalues of the
            spatial hessian matrix to reject edge responses

    Returns:
        (cv2.KeyPoint): a keypoint after localization
        (int): the image index in the octave
    """
    is_outside_image = False
    im_shape = dog_images_in_octave[0].shape
    for iteration in range(max_iteration):
        im1, im2, im3 = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([im1[y-1:y+2, x-1:x+2],
                               im2[y-1:y+2, x-1:x+2],
                               im3[y-1:y+2, x-1:x+2]]).astype('float32') / 255
        gradient = gradient_at_center_pixel(pixel_cube)
        hessian = hessian_at_center_pixel(pixel_cube)
        extremum_offset = -1 * np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if np.all(np.abs(extremum_offset) < 0.5):
            break

        # update location based on the new estimator offset
        x += int(round(extremum_offset[0]))
        y += int(round(extremum_offset[1]))
        image_index += int(round(extremum_offset[2]))

        # check new position is inside the image
        if not ((dist_from_boarder <= y <= im_shape[0] - dist_from_boarder)
                and (dist_from_boarder <= x <= im_shape[1] - dist_from_boarder)
                and 1 <= image_index <= num_intervals):
            is_outside_image = True
            break
    else:  # finished the max iteration but not converge
        return None

    if is_outside_image:
        return None
    value_at_extremum_estimator = pixel_cube[1, 1, 1] + 0.5 * gradient @ extremum_offset
    if np.abs(value_at_extremum_estimator) < 0.03:  # discard extrema with |D(x_hat)| < 0.03
        return None
    # Eliminate edge response
    spatial_hessian = hessian[:2, :2]
    spatial_hessian_trace = np.trace(spatial_hessian)
    spatial_hessian_det = np.linalg.det(spatial_hessian)
    if spatial_hessian_det > 0 and spatial_hessian_trace ** 2 / spatial_hessian_det < (eigenvalue_ratio + 1) ** 2 / eigenvalue_ratio:
        keypoint = cv2.KeyPoint()
        keypoint.pt = ((x + extremum_offset[0]) * 2 ** octave_index, (y + extremum_offset[1]) * 2 ** octave_index)
        # the following properties are according to OpenCV:
        # the octave property encode the octave itself and the image index in the octave
        keypoint.octave = octave_index \
                          + image_index * 2 ** 8
        keypoint.size = sigma * (2 ** ((image_index + extremum_offset[2]) / float(num_intervals))) * (
                    2 ** (octave_index + 1))
        keypoint.response = np.abs(value_at_extremum_estimator)
        return keypoint, image_index


def add_orientation_to_key_point(keypoint, octave_index, blurred_image, radius_factor=3, num_bins=36, peak_ratio=0.8,
                                 scale_factor=1.5):
    """
    Add to each keypoint the orientation of its main gradients. If there
    is more than one dominant orientation, the original keypoint will
    be duplicated with each of the dominant orientation.
    Args:
        keypoint (cv2.KeyPoint): The keypoint to assign orientation to
        octave_index (int): The level in the pyramid the keypoint presents
        blurred_image (np.ndarray): the blurred image where the keypoint presents
        radius_factor (int): The base radius of the neighbourhood to calculate
            the gradients orientations around the keypoint.
        num_bins (int): The number of bins in the orientation histogram
        peak_ratio (float): The ratio between a peak in the histogram and
            the highest peak in the histogram that above it the peak is
            also consider an orientation of the keypoint.
        scale_factor (float): a factor to construct the gaussian weighting
            of the gradients.

    Returns:
        (list[cv2.KeyPoint]): a list with the original keypoint after added
            orientations. The keypoint will be duplicated for each orientation.
    """

    keypoints = []
    image_shpae = blurred_image.shape

    sigma = scale_factor * keypoint.size / float(2 ** (octave_index + 1))
    radius = int(round(radius_factor * sigma))
    gaussian_weight_factor = -0.5 * 1 / (sigma ** 2)

    histogram = np.zeros(num_bins)

    y = keypoint.pt[1]
    x = keypoint.pt[0]
    scaled_y = int(np.round(y / (2 ** octave_index)))
    scaled_x = int(np.round(x / (2 ** octave_index)))

    for i in range(scaled_y - radius, scaled_y + radius + 1):
        if i <= 0 or image_shpae[0] - 1 <= i:
            continue
        for j in range(scaled_x - radius, scaled_x + radius + 1):
            if j <= 0 or image_shpae[1] - 1 <= j:
                continue
            dx = blurred_image[i, j + 1] - blurred_image[i, j - 1]
            dy = blurred_image[i - 1, j] - blurred_image[i + 1, j]  # positive direction is upward
            gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
            gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
            weight = np.exp(gaussian_weight_factor * ((i - scaled_y) ** 2 + (j - scaled_x) ** 2))
            histogram_bin = int(np.round(num_bins * gradient_orientation / 360)) % num_bins
            histogram[histogram_bin] += weight * gradient_magnitude

    max_value_in_hist = np.max(histogram)
    high_peaks_positions = np.nonzero(
        (histogram > np.roll(histogram, 1)) & (histogram > np.roll(histogram, -1))
        & (histogram >= peak_ratio * max_value_in_hist))[0]
    for peak_index in high_peaks_positions:
        peak_value = histogram[peak_index]
        left_value = histogram[peak_index - 1]
        right_value = histogram[(peak_index + 1) % num_bins]
        # interpolation is given by (6.30) at https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
        orientation = interpolated_peak_index * 360 / num_bins
        if np.abs(orientation) < float_tolerance:
            orientation = 0
        keypoint_with_orientation = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
        keypoints.append(keypoint_with_orientation)
    return keypoints


def compare_keypoints(keypoint_1, keypoint_2):
    """
    Function to compare 2 keypoints. The value is positive if keypoint_1
        is considered greater than keypoint_2 and negative if keypoint_2
        is greater. The function return 0 if the keypoints are equal.
    Args:
        keypoint_1 (cv2.KeyPoint): keypoint to compare
        keypoint_2 (cv2.KeyPoint): keypoint to compare

    Returns:
        (float): A positive number if keypoint_1 > keypoint_2, negative if
            keypoint_2 > keypoint_1, and 0 if keypoint_1 == keypoint_2
    """
    if keypoint_1.pt[0] != keypoint_2.pt[0]:
        return keypoint_1.pt[0] - keypoint_2.pt[0]
    if keypoint_1.pt[1] != keypoint_2.pt[1]:
        return keypoint_1.pt[1] - keypoint_2.pt[1]
    if keypoint_1.size != keypoint_2.size:
        return keypoint_2.size - keypoint_1.size
    if keypoint_1.angle != keypoint_2.angle:
        return keypoint_1.angle - keypoint_2.angle
    if keypoint_1.response != keypoint_2.response:
        return keypoint_2.response - keypoint_1.response
    if keypoint_1.octave != keypoint_2.octave:
        return keypoint_2.octave - keypoint_1.octave
    return 0


def get_unique_keypoints(keypoints):
    """
    Remove duplicate keypoints from the list, and return a list of unique
    keypoints.
    Args:
        keypoints (list[cv2.KeyPoint]): a list of keypoints

    Returns:
        (list[cv2.KeyPoint]): a list with unique keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    for keypoint in keypoints[1:]:
        if compare_keypoints(unique_keypoints[-1], keypoint) != 0:
            unique_keypoints.append(keypoint)
    return unique_keypoints


def convert_keypoint_realtive_to_original_image(keypoints):
    """
        Convert the keypoint properties relative to original (input)
        image (which is twice smaller than the base image). That means,
        the keypoint properties are as the base of the pyramid is the
        input image, instead of the base image.
    Args:
        keypoints (list[cv2.KeyPoint]): a list of keypoints

    Returns:
        (list[cv2.KeyPoint]): list of keypoints after conversion to the
            original image
    """
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        # go one octave down, and save the other encoding as it
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
    return keypoints


def decode_octave(keypoint):
    """
    The property octave contains 3 elements. 1. The octave itself
    (meaning the level in the pyramid), 2. the layer inside the
    octave and 3. from the octave we can deduce the scale of the
    image. After converting the keypoint to the original (input)
    image, the octave can be negative.
    Args:
        keypoint (cv2.KeyPoint): a keypoint object

    Returns:
        (int): the octave (level in the pyramid) of the keypoint
        (int): the layer of the keypoint in the octave
        (float): the scale of the image of the keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:  # meaning octave below the original image
        octave = octave | -128
    scale = 1 / float(2 ** octave) if octave >= 0 else float(2 ** -octave)  # scale relative to the original image
    return octave, layer, scale


def generate_descriptor(keypoints, blurred_images, descriptor_max_value=0.2):
    """
    Create a descriptor for each keypoint
    Args:
        keypoints (list[cv2.KeyPoint]): a list with the keypoints to make descriptor for
        blurred_images (list): a list contains the blurred images in every level
        descriptor_max_value (float): The maximum value a descriptor can hold

    Returns:
        (np.ndarray): a descriptors matrix. Each descriptor is a row in the matrix
    """
    orientation_num_of_bins = 8
    spatial_num_of_bins = 4
    window_size = 16
    descriptors = []
    bins_per_degree = orientation_num_of_bins / 360
    bins_per_pixel = spatial_num_of_bins / window_size
    gaussian_kernel = cv2.getGaussianKernel(window_size, window_size / 2)
    weight = gaussian_kernel @ gaussian_kernel.T

    for keypoint in keypoints:
        octave, layer, scale = decode_octave(keypoint)
        blurred_image = blurred_images[octave + 1][layer]
        rows_num, cols_num = blurred_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        angle = keypoint.angle
        row_bins = []
        col_bins = []
        magnitude = []
        orientation_bin = []
        histogram = np.zeros((spatial_num_of_bins + 2, spatial_num_of_bins + 2, orientation_num_of_bins))
        for delta_x in range(-1 * window_size // 2, window_size // 2):
            for delta_y in range(-1 * window_size // 2, window_size // 2):
                x = point[0] + delta_x
                y = point[1] + delta_y
                if 1 <= x < cols_num - 1 and 1 <= y < rows_num - 1:
                    dx = blurred_image[y, x + 1] - blurred_image[y, x - 1]
                    dy = blurred_image[y - 1, x] - blurred_image[y + 1, x]
                    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                    row_bins.append((window_size / 2 + delta_y) * bins_per_pixel - 0.5)
                    col_bins.append((window_size / 2 + delta_x) * bins_per_pixel - 0.5)
                    magnitude.append(weight[window_size // 2 + delta_y, window_size // 2 + delta_x] * gradient_magnitude)
                    orientation_bin.append((gradient_orientation - angle) * bins_per_degree)

        for i_bin, j_bin, magnitude, orientation_bin in zip(row_bins, col_bins, magnitude, orientation_bin):
            i_bin_floor = np.floor(i_bin).astype('int')
            j_bin_floor = np.floor(j_bin).astype('int')
            orientation_bin_floor = np.floor(orientation_bin).astype('int')
            i_fraction = i_bin - i_bin_floor
            j_fraction = j_bin - j_bin_floor
            orientation_fraction = orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += orientation_num_of_bins
            if orientation_bin_floor >= orientation_num_of_bins:
                orientation_bin_floor -= orientation_num_of_bins

            c0 = magnitude * (1 - i_fraction)
            c1 = magnitude * i_fraction
            c00 = c0 * (1 - j_fraction)
            c01 = c0 * j_fraction
            c10 = c1 * (1 - j_fraction)
            c11 = c1 * j_fraction
            c000 = c00 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c111 = c11 * orientation_fraction

            histogram[i_bin_floor + 1, j_bin_floor + 1, orientation_bin_floor] += c000
            histogram[i_bin_floor + 1, j_bin_floor + 1, (orientation_bin_floor + 1) % orientation_num_of_bins] += c001
            histogram[i_bin_floor + 1, j_bin_floor + 2, orientation_bin_floor] += c010
            histogram[i_bin_floor + 1, j_bin_floor + 2, (orientation_bin_floor + 1) % orientation_num_of_bins] += c011
            histogram[i_bin_floor + 2, j_bin_floor + 1, orientation_bin_floor] += c100
            histogram[i_bin_floor + 2, j_bin_floor + 1, (orientation_bin_floor + 1) % orientation_num_of_bins] += c101
            histogram[i_bin_floor + 2, j_bin_floor + 2, orientation_bin_floor] += c110
            histogram[i_bin_floor + 2, j_bin_floor + 2, (orientation_bin_floor + 1) % orientation_num_of_bins] += c111

        descriptor = (histogram[1:-1, 1:-1, :]).flatten()
        # threshold and normalize
        normalized_descriptor = descriptor / np.linalg.norm(descriptor)
        clipped_normalized_descriptor = np.clip(normalized_descriptor, None, descriptor_max_value)
        renormalized_descriptor = clipped_normalized_descriptor / max(np.linalg.norm(clipped_normalized_descriptor), float_tolerance)

        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor = np.round(512 * renormalized_descriptor)
        descriptor = np.clip(descriptor, 0, 255)
        descriptors.append(descriptor)
    return np.array(descriptors, dtype='float32')


def get_keypoints_and_descriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """
    The main function which generate the keypoints and the descriptor for an image
    Args:
        image (np.ndarray): The image to make descriptors for
        sigma (float): the sigma for the blurring of the base image in the pyramid
        num_intervals (int): number of interval of blurred images in an octave
        assumed_blur (float): The blur we assume the original image has
        image_border_width (int): The number of pixels to keep from the
            image's boarder when searching for extrema

    Returns:
        (list[cv2.KeyPoint]): list of keypoints found in the image
        (np.ndarray): array of descriptors for each keypoint
    """
    image = image.astype('float32')
    base_image = generate_base_pyramid_image(image, sigma, assumed_blur)
    num_octaves = get_number_of_pyramid_levels(base_image)
    gaussian_kernels = get_octave_gaussian_sigma(sigma, num_intervals)
    blurred_images = generate_scale_space_pyramid(base_image, num_octaves, gaussian_kernels)
    dog_images = generate_dog_pyramid(blurred_images)
    keypoints = scale_space_extrema_detection(dog_images, blurred_images, num_intervals, image_border_width, sigma)
    keypoints = get_unique_keypoints(keypoints)
    keypoints = convert_keypoint_realtive_to_original_image(keypoints)
    descriptors = generate_descriptor(keypoints, blurred_images)
    return keypoints, descriptors


if __name__ == '__main__':
    from random import randint
    import matplotlib.pyplot as plt
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('box.png', 0)
    img2 = cv2.imread('box_in_scene.png', 0)

    # get the keypoints and descriptors
    keypoints_1, descriptors_1 = get_keypoints_and_descriptors(img1)
    keypoints_2, descriptors_2 = get_keypoints_and_descriptors(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected boxing in scene image
        h, w = img1.shape
        corner_points = np.float32([[0, 0],
                                    [0, h - 1],
                                    [w - 1, h - 1],
                                    [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(corner_points, M)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        n_width = w1 + w2
        n_height = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((n_height, n_width, 3), np.uint8)

        newimg[hdif:hdif + h1, :w1, :] = img1
        newimg[:h2, w1:w1 + w2, :] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(keypoints_1[m.queryIdx].pt[0]), int(keypoints_1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(keypoints_2[m.trainIdx].pt[0] + w1), int(keypoints_2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, tuple([randint(0, 255) for i in range(3)]))

        plt.imshow(newimg)
        plt.show()
    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
