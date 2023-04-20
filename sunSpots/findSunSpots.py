from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from ImageRequest import ImageRequest


def iou(box1, box2):
    x1, y1, x2, y2, _ = box1
    x3, y3, x4, y4, _ = box2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    w_inter = max(0, x_inter2 - x_inter1)
    h_inter = max(0, y_inter2 - y_inter1)
    area_inter = w_inter * h_inter

    h_box1 = y2 - y1
    w_box1 = x2 - x1
    h_box2 = y4 - y3
    w_box2 = x4 - x3
    area_box1 = h_box1 * w_box1
    area_box2 = h_box2 * w_box2

    area_union = area_box1 + area_box2 - area_inter
    iou_val = area_inter / area_union
    return iou_val


def non_maximum_suppression(boxs: list):
    boxs = sorted(boxs, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    max_boxs = []
    while boxs:
        current = boxs.pop(-1)
        max_boxs.append(current)
        index_to_remove = []
        for ii, b in enumerate(boxs):
            iou_val = iou(current, b)
            if iou_val > 0.12:
                index_to_remove.append(ii)
        boxs = [x for ii, x in enumerate(boxs) if ii not in index_to_remove]
    return max_boxs


def find_sunspot(image):
    im_edges = cv2.Canny(image, 270, 350)
    im_edges = cv2.dilate(im_edges, None, iterations=3)
    im_edges = cv2.erode(im_edges, None, iterations=3)
    contours, hierarchy = cv2.findContours(im_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)
    max_contour_index, max_contour = max(enumerate(contours), key=lambda tup: cv2.contourArea(tup[1]))
    max_contour_first_child = int(hierarchy[max_contour_index, 2])
    moments = cv2.moments(max_contour)
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    radius = np.mean(np.sqrt((max_contour[:, :, 0] - center_x) ** 2 + (max_contour[:, :, 1] - center_y) ** 2))
    pixel_size_km = 696_340 / radius

    spots_contour = itemgetter(*np.flatnonzero(hierarchy[:, 3] == max_contour_first_child))(contours)

    im_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    boxs = []
    oversize = 2
    for cnt in spots_contour:
        x, y, w, h = cv2.boundingRect(cnt)
        _, radius = cv2.minEnclosingCircle(cnt)
        boxs.append((x - oversize, y - oversize, x + w + oversize, y + h + oversize, radius))

    boxs = non_maximum_suppression(boxs)

    for x1, y1, x2, y2, r in boxs:
        cv2.rectangle(im_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for x1, y1, x2, y2, r in boxs:
        cv2.putText(im_rgb, f'{int(pixel_size_km * r)}km', (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                    cv2.LINE_AA)
    cv2.putText(im_rgb, f'{len(boxs)}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 0, 0), 10, cv2.LINE_AA)
    return im_rgb


def find_spots_adaptive(image):
    pixel_size_km = 363.313
    image_blur = cv2.GaussianBlur(image, (13, 13), 0)
    binary_img = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 301, 30)
    # kernel = np.ones((5, 5), np.uint8)
    # binary_img = cv2.erode(binary_img, kernel, iterations=2)
    # binary_img = cv2.dilate(binary_img, kernel, iterations=2)
    # binary_img = cv2.dilate(binary_img, None, iterations=2)
    # binary_img = cv2.erode(binary_img, None, iterations=2)

    binary_img = cv2.dilate(binary_img, None, iterations=1)
    binary_img = cv2.erode(binary_img, None, iterations=1)



    _, _, boxes, centroid = cv2.connectedComponentsWithStats(binary_img)
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    filtered_centroid = []
    for (x, y, w, h, pixels), cent in zip(boxes, centroid):
        if pixels < 10000 and h > 10 and w > 10:
            filtered_boxes.append((x, y, w, h))
            filtered_centroid.append(cent)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    oversize = 2
    for x, y, w, h in filtered_boxes:
        cv2.rectangle(im_rgb, (x - oversize, y - oversize), (x + w + oversize, y + h + oversize), (255, 0, 0), 1)
    for (x, y, w, h), (x_cent, y_cent) in zip(filtered_boxes, filtered_centroid):
        r = max(w / 2, h / 2)#np.sqrt((x - x_cent) ** 2 + (y - y_cent) ** 2)
        cv2.putText(im_rgb, f'{int(pixel_size_km * r)}km', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                    cv2.LINE_AA)
    cv2.putText(im_rgb, f'{len(filtered_boxes)}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 0, 0), 10, cv2.LINE_AA)
    return im_rgb


if __name__ == '__main__':
    date = (17, 7, 2022)
    im_request = ImageRequest()
    im = im_request.get_image(*date)
    del im_request

    test_im = find_spots_adaptive(im)
    plt.imshow(test_im)