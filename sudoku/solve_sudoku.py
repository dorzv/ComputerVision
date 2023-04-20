import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
# from digit_recognition import get_model
from printed_digit_recognition import get_model
from sudoku import Sudoku
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = get_model()
model.load_state_dict(torch.load('data/model_printed.pt'))
model.eval()


def get_number(im):
    """
    predict digit from image
    Args:
        im (np.ndarray): image of digit

    Returns:
        int: the digit that was predicted
    """
    im = torch.tensor(im / 255).float().to(device)
    im = im.view(-1, 1, 28, 28)
    prediction = model(im)
    max_values, argmaxes = prediction.max(-1)
    return int(argmaxes.cpu())


def imshow(im):
    """
    Show image. Can show gray-scale image or RGB
    Args:
        im (np.ndarray): image to be shown. Can be of 2-D (gray-scale) or 3-D (RGB)

    Returns:

    """
    plt.figure()
    if im.ndim == 2:
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(im)


# im_orig = cv2.imread('test_image.jpeg', 0)
im_orig = cv2.imread('sudoku_real_2.jpeg', 0)

# blur and thershold
im = cv2.GaussianBlur(im_orig, (7, 7), 0)
im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)

# invert
im = cv2.bitwise_not(im)
im = cv2.dilate(im, np.ones((3, 3), np.uint8), iterations=1)

num_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(im)
max_label = max(range(1, num_of_labels), key=lambda x: stats[x, 4])
outer_grid = np.zeros(im.shape, np.uint8)
outer_grid[labels == max_label] = 255
outer_grid = cv2.erode(outer_grid, None, iterations=1)

lines = cv2.HoughLines(outer_grid, 1, np.pi / 180, 150).squeeze()

# upper line
line_reduced = lines[(np.deg2rad(60) < lines[:, 1]) & (lines[:, 1] < np.deg2rad(120)), :]
r_up, theta_up = line_reduced[np.argmin(np.abs(line_reduced[:, 0])), :]
x_up = r_up * np.cos(theta_up)
y_up = r_up * np.sin(theta_up)

# bottom line
r_down, theta_down = line_reduced[np.argmax(np.abs(line_reduced[:, 0])), :]

# left most line
line_reduced = lines[((0 <= lines[:, 1]) & (lines[:, 1] < np.deg2rad(30)))
                     | ((np.deg2rad(160) <= lines[:, 1]) & (lines[:, 1] <= np.deg2rad(180))), :]
r_left, theta_left = line_reduced[np.argmin(np.abs(line_reduced[:, 0])), :]

# right most line
r_right, theta_right = line_reduced[np.argmax(np.abs(line_reduced[:, 0])), :]

# find the intersection points (corners)
# base on the relation x = x0 + s * n_x, y = y0 + s * n_y (4 equation with 4 variables
corners = []
for (r1, theta1), (r2, theta2) in [[(r_up, theta_up), (r_left, theta_left)], [(r_up, theta_up), (r_right, theta_right)],
                                   [(r_down, theta_down), (r_right, theta_right)],
                                   [(r_down, theta_down), (r_left, theta_left)]]:
    x_01 = r1 * np.cos(theta1)
    y_01 = r1 * np.sin(theta1)
    x_02 = r2 * np.cos(theta2)
    y_02 = r2 * np.sin(theta2)
    n_x1 = -np.sin(theta1)
    n_y1 = np.cos(theta1)
    n_x2 = -np.sin(theta2)
    n_y2 = np.cos(theta2)
    A = np.array([[1, 0, -n_x1, 0], [1, 0, 0, -n_x2], [0, 1, -n_y1, 0], [0, 1, 0, -n_y2]])
    b = np.array([x_01, x_02, y_01, y_02])
    x, y = np.linalg.solve(A, b)[:2]
    corners.append((int(np.round(x)), int(np.round(y))))

# draw the lines on the original image
# im_with_lines = im_orig.copy()
# for ii, (r, theta) in enumerate([(r_up, theta_up), (r_down, theta_down), (r_left, theta_left), (r_right, theta_right)]):
#     # component of the direction of the line
#     n_x = -np.sin(theta)
#     n_y = np.cos(theta)
#     x0 = r * np.cos(theta)
#     y0 = r * np.sin(theta)
#     x1 = int(x0 + 1000 * n_x)
#     y1 = int(y0 + 1000 * n_y)
#     x2 = int(x0 - 1000 * n_x)
#     y2 = int(y0 - 1000 * n_y)
#     cv2.line(im_with_lines, (x1, y1), (x2, y2), 255, 1)
#     cv2.circle(im_with_lines, corners[ii], 7, 0, -1)

# correct perspective
width_up = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
width_down = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
height_right = np.sqrt((corners[1][0] - corners[2][0]) ** 2 + (corners[1][1] - corners[2][1]) ** 2)
height_left = np.sqrt((corners[0][0] - corners[3][0]) ** 2 + (corners[0][1] - corners[3][1]) ** 2)

max_width = max(int(width_up), int(width_down))
max_height = max(int(height_right), int(height_left))

dst = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]
], np.float32)

M = cv2.getPerspectiveTransform(np.array(corners, np.float32), dst)
warp = cv2.warpPerspective(im_orig, M, (max_width, max_height))

# blur and thershold
im = cv2.GaussianBlur(warp, (9, 9), 0)
im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

# invert
im = cv2.bitwise_not(im)
# imshow(im)

step_x = im.shape[1] // 9
step_y = im.shape[0] // 9
sudoku_mat = np.zeros((9, 9), int)
for ii in range(9):
    for jj in range(9):
        cell = im[max(ii*step_y - 2, 0):min((ii+1)*step_y + 2, im.shape[0]),
                  max(jj*step_x - 2, 0):min((jj+1)*step_x + 2, im.shape[1])]
        if np.sum(cell == 255) < 0.1 * cell.size:
            continue
        _, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cell)
        cell_center = np.array([int(cell.shape[1]/2), int(cell.shape[0]/2)])

        if np.all(labels_im[int(cell_center[1]-cell.shape[1]*0.125):int(cell_center[1]+cell.shape[1]*0.125),
                  int(cell_center[0]-cell.shape[0]*0.125):int(cell_center[0]+cell.shape[0]*0.125)] == 0):
            continue
        stats = stats[1:]
        centroids = centroids[1:]
        inx_sort = np.argsort(stats[:, 4])[::-1]
        stats = stats[inx_sort, :]
        centroids = centroids[inx_sort, :]

        for kk in range(min(2, stats.shape[0])):
            if stats[kk, 4] < 0.05 * cell.size:
                break
            if stats[kk, 2] > 0.8 * cell.shape[1] or stats[kk, 3] > 0.8 * cell.shape[0]:
                continue
            if ((0.25 * cell.shape[1]) < centroids[kk, 0] < (0.75 * cell.shape[1])
                and (0.25 * cell.shape[0]) < centroids[kk, 1] < (0.75 * cell.shape[0])):
                digit = np.zeros((28, 28), np.uint8)
                x, y, w, h = stats[kk, :4]
                if w < 28 and h < 28:
                    digit[int(28/2 - h/2):int(28/2 + h/2),
                          int(28/2 - w/2):int(28/2 + w/2)] = cell[y:y+h, x:x+w].copy()
                else:  # assume h > w
                    ratio = 26 / h
                    new_width = int(ratio * w)
                    new_height = int(ratio * h)
                    resized_im = cv2.resize(cell[y:y+h, x:x+w], (new_width, new_height))
                    digit[int(28 / 2 - new_height / 2):int(28 / 2 + new_height / 2),
                          int(28 / 2 - new_width / 2):int(28 / 2 + new_width / 2)] = resized_im
                digit = cv2.GaussianBlur(digit, (5, 5), 0.5)
                sudoku_mat[ii, jj] = get_number(digit)  #get_number(cv2.GaussianBlur(digit, (5,5), 0.5))
                break


puzzle = Sudoku(3, 3, board=sudoku_mat.tolist())
puzzle.show()
puzzle.solve().show_full()

