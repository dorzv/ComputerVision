import numpy as np
import matplotlib.pyplot as plt
import cv2


def mean_shift(probability_image, roi_window, max_iteration=10, eps=2):
    """
    Use the mean shift algorithm to track an object. We assume the bounding box is moving but do not change
    its size or orientation.
    Args:
        probability_image (np.array): 2-D matrix with the size of the frame, with values that are the scores (or
            probability) for each pixel to belong to the object being tracked according to the pixels values histogram
            of the original bounding box of the tracked object
        roi_window (Tuple[int, int, int, int]): the bounding box of the tracked object from the previous frame as
            (x_min, y_min, width, height).
        max_iteration (int): maximum number of iteration to apply a mean shift step
        eps (float): the value of mean shift distance that considered as convergence of the algorithm (if the
            step distance of the mean shift is smaller than that value, the algorithm stops)

    Returns:
        (int): x value of the top left corner of the tracked object bounding box
        (int): y value of the top left corner of the tracked object bounding box
        (int): the width of the tracked object bounding box
        (int): the height of the tracked object bounding box
    """
    x, y, w, h = roi_window
    for ii in range(max_iteration):
        # calculate the moments of the probability image in the region of interest
        moments = cv2.moments(probability_image[y:y+h, x:x+w])

        # find the position of the mean (center of mass) in the ROI
        x_mean = moments['m10'] / moments['m00']
        y_mean = moments['m01'] / moments['m00']

        # set an updated value for the top right corner of the bounding box according to the new mean position
        new_x = np.round(x + x_mean - 0.5 * w)
        new_y = np.round(y + y_mean - 0.5 * h)

        # if the bounding box goes outside the frame, clap it to fit in
        if new_x < 0:
            new_x = 0
        elif probability_image.shape[1] - 1 < x_mean:
            new_x = probability_image[1] - 1 - w

        if new_y < 0:
            new_y = 0
        elif probability_image.shape[0] - 1 < y_mean:
            new_y = probability_image[0] - 1 - h

        # find the step size in each direction
        dx = new_x - x
        dy = new_y - y

        # set integer values to the updated coordinate of the top right corner
        x = int(new_x)
        y = int(new_y)

        # if the step size is smaller than eps we can stop
        if np.sqrt(dx ** 2 + dy ** 2) < eps:
            break

    return x, y, w, h


def camshift(probability_image, roi_window, with_orientation=False, max_iteration=10, eps=2):
    """
    Use the cam shift algorithm to track an object. In the default mode it applies only the mean shift algorithm.
    Args:
        probability_image (np.array): 2-D matrix with the size of the frame, with values that are the scores (or
            probability) for each pixel to belong to the object being tracked according to the pixels values histogram
            of the original bounding box of the tracked object
        roi_window (Tuple[int, int, int, int]): the bounding box of the tracked object from the previous frame as
            (x_min, y_min, width, height).
        with_orientation (bool): if true, apply the cam shift algorithm including update the orientation and size of the
            bounding box
        max_iteration (int): maximum number of iteration to apply a mean shift step
        eps (float): the value of mean shift distance that considered as convergence of the algorithm (if the
            step distance of the mean shift is smaller than that value, the algorithm stops)

    Returns:
        (float): x-coordinate of the center of the new bounding box
        (float): y-coordinate of the center of the new bounding box
        (int): the new width of the bounding box
        (int): the new height of the bounding box
        (float): the angle of the orientation of the bounding box
    """
    # apply the mean shift algorithm and get updated bounding box
    x, y, w, h = mean_shift(probability_image, roi_window, max_iteration, eps)

    # center coordinate of the bounding box
    xc = x + 0.5 * w
    yc = y + 0.5 * h

    # compute the moments of the new bounding box
    moments = cv2.moments(probability_image[y:y + h, x:x + w])

    # if with_orientation, get the angle of the orientation of the bounding box
    # according to the steps in the original paper
    if with_orientation:
        a = moments['m20'] / moments['m00'] - xc ** 2
        b = 2 * (moments['m11'] / moments['m00'] - xc * yc)
        c = moments['m02'] / moments['m00'] - yc ** 2
        theta = np.rad2deg(0.5 * np.arctan(b / (a - c)))
    else:
        theta = 0

    # calculate the new height (and width) according to method describe in the paper
    s = 2 * np.sqrt(moments['m00'] / 256)

    return xc, yc, int(1.5 * s), int(s), theta

# create video capture object and read the first frame
cap = cv2.VideoCapture('motorcycle.mp4')
ret, frame = cap.read()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
# initialize video writer object to make the result video
result = cv2.VideoWriter('motorcycle_tracked.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# corers of the bounding box of the object to track
x1 = 317  # left
y1 = 115  # top
x2 = 430  # right
y2 = 258  # bottom

w = x2 - x1  # width of the bounding box
h = y2 - y1  # height of the bounding box

roi = (x1, y1, w, h)  # region of interest bounding box

# show the bounding box of the object
im_marked = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
plt.figure()
plt.imshow(im_marked[..., ::-1])

im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert the colors from BGR to HSV
window_hsv = im_hsv[y1:y1+h, x1:x1+w]
# create a mask to include only HSV values that will be good for tracking
mask = cv2.inRange(window_hsv, np.array((0., 100., 100.)), np.array((180., 255., 255.)))
hist = cv2.calcHist([window_hsv], [0], mask, [180], [0, 180])  # get histogram of the Hue values of the pixels in the maks
hist_normalized = 255 * hist / hist.max()

# go over the next frames
while True:
    ret, frame = cap.read()
    if not ret:  # if there is no next frame to read, stop
        break
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # give each pixel a score according to the histogram
    prob_img = np.squeeze(np.round(hist_normalized[frame_hsv[:, :, 0]])).astype('uint8')
    # roi = mean_shift(prob_img, roi)
    # run the camshift algorithm
    xc, yc, w, h, theta = camshift(prob_img, roi, False)
    roi = np.round([xc - w/2, yc - h/2, w, h]).astype(int)

    # draw bounding box over the tracked object
    rect = cv2.boxPoints(((xc, yc), (w, h), theta)).astype(int)
    im_marked = cv2.drawContours(frame.copy(), [rect], 0, (0, 0, 255), 2)
    # add the new frame with bounding box to the result video
    result.write(im_marked)

# release video objects
cap.release()
result.release()
