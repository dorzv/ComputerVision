import matplotlib.pyplot as plt
import numpy as np
import cv2

def compute_weight(p, q):
    """
    Compute the weight based on Bhattacharyya coefficient between 2 normalized histograms
    Args:
        p (np.ndarray): normalized histogram
        q (np.ndarray): normalized histogram

    Returns:
        (float): weight based on Bhattacharyya coefficient between p and q
    """
    # make sure the histogram is normalized
    if not np.isclose(np.sum(p), 1):
        raise ValueError('p histogram is not normalized')
    if not np.isclose(np.sum(q), 1):
        raise ValueError('q histogram is not normalized')
    bc = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
    return np.exp(20 * bc)

def compute_norm_hist(image, state):
    """
    Compute histogram based on the RGB values of the image in the ROI defined by the state
    Args:
        image (np.ndarray): matrix represent image with the object to track in it
        state (np.ndarray): vector of the state of the tracked object

    Returns:
        vector of the normalized histogram of the colores in the ROI of the image based on the state
    """
    # get the top-left and bottom-right corner of the object bounding box
    # if the bounding box is outside the frame, we clap it
    x_min = max(np.round(state[0] - half_width).astype(int), 0)
    x_max = min(np.round(state[0] + half_width).astype(int), image.shape[1])
    y_min = max(np.round(state[1] - half_height).astype(int), 0)
    y_max = min(np.round(state[1] + half_height).astype(int), image.shape[0])

    roi = image[y_min:y_max+1, x_min:x_max+1]
    roi_reduced = roi // 16  # change range of pixel values from 0-255 to 0-15
    # build vector to represent the combinations of RGB as single value
    roi_indexing = (roi_reduced[..., 0] + roi_reduced[..., 1] * 16 + roi_reduced[..., 2] * 16 ** 2).flatten()
    hist, _ = np.histogram(roi_indexing, bins=4096, range=(0,4096))
    norm_hist = hist / np.sum(hist)  # normalize the histogram
    return norm_hist

def predict_particles(particles_states):
    """
    Predict new particles (states) based on the previous states
    Args:
        particles_states (np.ndarray): matrix of states. The rows are the properties of the state and the columns are the particles

    Returns:
        (np.ndarray): new predicted particles
    """
    # the dynamic we use assume that:
    # x_new = x + v_x (x - center position)
    # y_new = y + v_y (y - center position)
    # v_x_new = v_x (velocity at x)
    # v_y_new = v_y (velocity at y)
    dynamics = np.array(
        [[1 ,0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    # multiplying the state by the dynamics and add some noise
    predicted_states = dynamics @ particles_states + np.random.normal(loc=0, scale=1, size=particles_states.shape)
    return predicted_states

def sample_particle(particles_states, weights):
    """
    Sample the particles (states) according to their weights.
    First a random numbers from uniform distribution U~[0,1] are generated.
    The random numbers are subtracted from the cumulative sum of the weights,
    and the sampling is made by selecting the state which its cumulative weights
    sum is the minimal one which is greater than the random number. If c[i] is the
    cumulative sum of the weights for the i-th state and r is a random number, the
    state which will be sampled is the j-th state so c[j] >= r and also c[j] is
    the smaller one that hold this inequality.
    Args:
        particles_states (np.ndarray): matrix which its columns are the states
        weights (np.ndarray): weights of each state

    Returns:
        (np.ndarray): matrix of sampled states according to their weights
    """
    normalized_weights = weights / np.sum(weights)
    sampling_weights = np.cumsum(normalized_weights)
    rand_numbers = np.random.random(sampling_weights.size)  # get random numbers from U~[0,1]
    cross_diff = sampling_weights[None, :] - rand_numbers[:, None]  # subtract from each weight any of the random numbers
    cross_diff[cross_diff < 0] = np.inf  # remove negative results
    sampling = np.argmin(cross_diff, axis=1)
    sampled_particles = particles_states[:, sampling]  # sample the particles
    return sampled_particles

def draw_bounding_box(image, states, weights, with_max=True):
    """
    Draw rectangle according to the weighted average state and (optionally) a rectangle of the state with
    the maximum score.
    Args:
        image (np.ndarray): the image to draw recangle on
        states (np.ndarray): the sampling states
        weights (np.ndarray): the weight of each state
        with_max (bool): True to draw a rectangle according to the state with the highest weight

    Returns:
        (np.ndarray): image with rectangle drawn on for the tracked object
    """
    mean_box = np.average(states, axis=1, weights=weights)
    x_c_mean, y_c_mean = np.round(mean_box[:2]).astype(int)
    image_with_boxes = image.copy()
    cv2.rectangle(image_with_boxes, (x_c_mean - half_width, y_c_mean - half_height),
                                     (x_c_mean + half_width, y_c_mean + half_height), (0, 255, 0), 1)
    if with_max:
        max_box = states[:, np.argmax(weights)]
        x_c_max, y_c_max = np.round(max_box[:2]).astype(int)
        cv2.rectangle(image_with_boxes, (x_c_max - half_width, y_c_max - half_height),
                                         (x_c_max + half_width, y_c_max + half_height), (0, 0, 255), 1)
    return image_with_boxes

def draw_particles(image, states, weights):
    """
    Draw the particles (position state) on the image with size according to the weight of the state.
    Args:
        image (np.ndarray): the image to draw particles on
        states (np.ndarray): the sampling states
        weights (np.ndarray): the weight of each state

    Returns:
        (np.ndarray): image with particles drawn on
    """
    image_with_particles = image.copy()
    for s, w in zip(states.T, weights):
        x, y = np.round(s[:2]).astype(int)
        cv2.circle(image_with_particles, (x, y), int(round(30 * w)), (0, 0, 255), thickness=-1)
    return image_with_particles

# create video reader object and read te first frame
cap = cv2.VideoCapture('simpson.avi')
ret, image = cap.read()

num_of_particles = 250
# s_init = np.array([375, 198, 0, 0])  # initial state [x_center, y_center, x_velocity, y_velocity]
s_init = np.array([81, 169, 0, 0])  # initial state [x_center, y_center, x_velocity, y_velocity]

# bounding box's (half) height and width.
# We assume those are constant
half_height = 118
half_width = 33

# get parameters of the video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20 #cap.get(5)  # frames per second

# create a video writer object
size = (frame_width, frame_height)
result = cv2.VideoWriter('simpson_tracked.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)
# get the normalized histogram of the initial state
# this histogram is the histogram we compare to later
q = compute_norm_hist(image, s_init)

# for the first frame all the particle are the same as the initial state
# and have the same weight since we don't have uncertainty in the state
s_new = np.tile(s_init, (num_of_particles, 1)).T
weights = np.ones(num_of_particles)

# go over the frames in the video
while True:
    # sample new particles according to the weights
    s_sampled = sample_particle(s_new, weights)
    # predict new particles (states) according to the previous states
    s_new = predict_particles(s_sampled)
    # go over the new predicted states, and compute the histogram for the state and
    # the weight with the original histogram
    for jj, s in enumerate(s_new.T):
        p = compute_norm_hist(image, s)
        weights[jj] = compute_weight(p, q)

    # draw bounding box over the tracked object
    image_with_boxes = draw_bounding_box(image, s_new, weights, with_max=True)
    # image_with_particles = draw_particles(image_with_boxes, s_sampled, normalized_weights)
    result.write(image_with_boxes)

    # read next frame
    ret, image = cap.read()
    if not ret:  # if there is no next frame to read, stop
        break

# release video objects
cap.release()
result.release()


