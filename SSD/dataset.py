import os
import json
from random import random, shuffle, uniform, randint, choice

import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT

from PIL import Image

from SSD.ssd_network import find_iou

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v for v, k in enumerate(voc_labels, 1)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

# bounding box colors for the different labels
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def parse_annotation(annotation_path):
    """
    Parse the xml annotation file of an image
    Args:
        annotation_path (str): path to annotation xml file

    Returns:
        (Dict['boxes': list, 'labels': list, 'difficulties': list): a dictionary with the keys
            boxes: a list with bounding boxes coordinate
            labels: a list with objects labels
            difficulties: a list with difficulties of objects
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []

    # go over all objects in the image
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bounding_box = object.find('bndbox')
        x_min = int(bounding_box.find('xmin').text) - 1
        y_min = int(bounding_box.find('ymin').text) - 1
        x_max = int(bounding_box.find('xmax').text) - 1
        y_max = int(bounding_box.find('ymax').text) - 1

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create json files with the information about the training data and testing data
    Args:
        voc07_path (str): path to the VOC2007 dataset
        voc12_path (str): path to the  VOC2012 dataset
        output_folder (str): path where the json file will be saved

    Returns:
        None
    """
    voc12_abs_path = os.path.abspath(voc12_path)
    voc07_abs_path = os.path.abspath(voc07_path)

    train_images = []
    train_objects = []
    num_of_objects = 0

    for path in [voc07_abs_path, voc12_abs_path]:
        # read the ids of the images belong to the training data
        with open(os.path.join(path, 'ImageSets', 'Main', 'trainval.txt'), 'r') as file:
            ids = file.read().splitlines()

        for id in ids:
            objects = parse_annotation(os.path.join(path, 'Annotations', f'{id}.xml'))  # get information for the id
            if len(objects['boxes']) == 0:  # if there is no object, continue
                continue
            num_of_objects += len(objects['labels'])  # count the number of objects
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', f'{id}.jpg'))

    # create json files with the information about the training images
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as json_file:
        json.dump(train_images, json_file)
    with open(os.path.join(output_folder, 'train_objects.json'), 'w') as json_file:
        json.dump(train_objects, json_file)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as json_file:
        json.dump(label_map, json_file)

    print(f'\nThere are {len(train_images)} training images containing a total of {num_of_objects} objects.'
          f' Files have been saved to {os.path.abspath(output_folder)}.')

    # repeat the previous steps for the testing data
    test_images = []
    test_objects = []
    num_of_objects = 0


    with open(os.path.join(voc07_abs_path, 'ImageSets', 'Main', 'test.txt'), 'r') as file:
        ids = file.read().splitlines()

    for id in ids:
        objects = parse_annotation(os.path.join(voc07_abs_path, 'Annotations', f'{id}.xml'))
        if len(objects['boxes']) == 0:
            continue
        num_of_objects += len(objects['labels'])
        test_objects.append(objects)
        test_images.append(os.path.join(voc07_abs_path, 'JPEGImages', f'{id}.jpg'))

    with open(os.path.join(output_folder, 'test_images.json'), 'w') as json_file:
        json.dump(train_images, json_file)
    with open(os.path.join(output_folder, 'test_objects.json'), 'w') as json_file:
        json.dump(train_objects, json_file)

    print(f'\nThere are {len(test_images)} training images containing a total of {num_of_objects} objects.'
          f' Files have been saved to {os.path.abspath(output_folder)}.')

def photometric_distort(image):
    """
    Apply photometric distortions including brightness, contrast, saturation and hue adjustment.
    The order of the transformations is random and each transformation has probability of 50%
    to be applied.
    Args:
        image (PIL.Image): image to apply transformation to

    Returns:
        (PIL.Image): image after transformations
    """
    transformed_image = image

    transformations = [FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation, FT.adjust_hue]
    shuffle(transformations)

    for t in transformations:
        if random() < 0.5:
            if t.__name__ == 'adjust_hue':
                adjust_factor = uniform(-18 / 255, 18 / 255)
            else:
                adjust_factor = uniform(0.5, 1.5)

            transformed_image = t(transformed_image, adjust_factor)
    return transformed_image

def expand(image, boxes, filler):
    """
    Expand the image (zoom out), so the image will be smaller on a canvas with filler value.
    Args:
        image (torch.tensor): image as tensor with dimension (3, h, w)
        boxes (torch.tensor): bounding boxes of the objects with coordinate (x_min, y_min, x_max, y_max),
            size (num objects, 4)
        filler (list): RGB filler value as [R, G, B]

    Returns:
        (torch.tensor): new expand image
        (torch.tensor): new bounding boxes coordinate after image expansion
    """
    # get new dimension of the expanded image
    max_scale = 4
    original_h = image.shape[1]
    original_w = image.shape[2]
    scale = uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # create the canvas
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * torch.tensor(filler, dtype=torch.float)[:, None, None]
    # randomly select the location of the original image in the canvas
    left = randint(0, new_w - original_w)
    right = left + original_w
    top = randint(0, new_h - original_h)
    bottom = top + original_h
    # place the original image in the canvas
    new_image[:, top:bottom, left:right] = image
    # update the location of the bounding boxes
    new_boxes = boxes + torch.tensor([left, top, left, top], dtype=torch.float)[None, :]

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Crop the image randomly into a smaller part of the original image
    Args:
        image (torch.tensor): image as tensor with dimension (3, h, w)
        boxes (torch.tensor): bounding boxes of the objects with coordinate (x_min, y_min, x_max, y_max),
            size (num objects, 4)
        labels (torch.tensor): labels of the objects in the image, tensor with dimension (num objects)
        difficulties (torch.tensor): the difficulties of the objects in the image, tensor with dimension (num objects)

    Returns:
        (torch.tensor): image after cropping
        (torch.tensor): new bounding boxes coordinate after image cropping
        (torch.tensor):  updated labels after image cropping
        (torch.tensor):  updated difficulties after image cropping
    """
    original_h = image.shape[1]
    original_w = image.shape[2]

    while True:
        # select randomly the minimum overlap
        min_overlap = choice([0, 0.1, 0.3, 0.5, 0.7, 0.9, None])
        # if None is selected, return the original image withput cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_trials = 50
        min_scale = 0.3
        for _ in range(max_trials):
            # get new dimensions for the image
            scale_h = uniform(min_scale, 1)
            scale_w = uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            aspect_ratio = new_h / new_w
            # new aspect ratio should be between 0.5 to 2
            if not 0.5 < aspect_ratio < 2:
                continue
            # get corners for the cropped image
            left = randint(0, original_w - new_w)
            right = left + new_w
            top = randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.tensor([left, top, right, bottom], dtype=torch.float)
            # find the overlap between the cropped part and the original boxes
            overlap = find_iou(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)
            # if the maximum overlap is less than the minimum overlap, try again
            if overlap.max().item() < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right]  # crop the image
            boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2  # find the centers of the original boxes
            # check if the center of the bounding boxes is within the cropped image
            centers_in_crop = (left < boxes_center[:, 0]) & (boxes_center[:, 0] < right) & (top < boxes_center[:, 1]) \
                & (boxes_center[:, 1] < bottom)
            # if none of the boxes centers is within the cropped image, try again
            if not torch.any(centers_in_crop):
                continue

            # keep only boxes with centers within the cropped image
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # update the boxes to lay inside the cropped image and update their coordinate
            new_boxes[:, :2] = torch.maximum(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.minimum(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

def flip(image, boxes):
    """
    Flip horizontally the image
    Args:
        image (PIL.Image): the image to be flipped
        boxes (torch.tensor): bounding boxes of the objects with coordinate (x_min, y_min, x_max, y_max),
            size (num objects, 4)

    Returns:
        (PIL.Image): image after transformations
        (torch.tensor): new bounding boxes coordinate after transformations
    """
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes

def resize(image, boxes, size=(300, 300), return_percent_coords=True):
    """
    Resize the image to the size the network except to get as input
    Args:
        image (PIL.Image): the image to be resized
        boxes (torch.tensor): bounding boxes of the objects with coordinate (x_min, y_min, x_max, y_max),
            size (num objects, 4)
        size (Tuple[int, int]): the new size of the image
        return_percent_coords (bool): True to return the updated box coordinate as percent

    Returns:
        (PIL.Image): image after resizing
        (torch.tensor): new bounding boxes coordinate after transformations
    """
    new_image = FT.resize(image, size)

    old_size = torch.tensor([image.width, image.height, image.width, image.height], dtype=torch.float).unsqueeze(0)
    new_boxes = boxes / old_size  # get the boxes coordinate as percent (i.e., normalized)

    if not return_percent_coords:
        new_size = torch.tensor([size[1], size[0], size[1], size[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_size  # get the boxes coordinate as pixels

    return new_image, new_boxes


def transform(image, boxes, labels, difficulties, data_type):
    """
    Apply transformation on an image
    Args:
        image (PIL.Image): the image to be transformed
        boxes (torch.tensor): bounding boxes of the objects with coordinate (x_min, y_min, x_max, y_max),
            size (num objects, 4)
        labels (torch.tensor): labels of the objects in the image, tensor with dimension (num objects)
        difficulties (torch.tensor): the difficulties of the objects in the image, tensor with dimension (num objects)
        data_type (str): either 'train' for training data or 'test' for testing data

    Returns:
        (torch.tensor): transformed image
        (torch.tensor): bounding boxes after transformation
        (torch.tensor): labels after transformation
        (torch.tensor): difficulties after transformation
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transformed_image = image
    transformed_boxes = boxes
    transformed_labels = labels
    transformed_difficulties = difficulties

    # part of transformation to be applied only on training data
    if data_type == 'train':
        transformed_image = photometric_distort(transformed_image)
        transformed_image = FT.to_tensor(transformed_image)
        if random() < 0.5:
            transformed_image, transformed_boxes = expand(transformed_image, boxes, filler=mean)
        transformed_image, transformed_boxes, transformed_labels, transformed_difficulties = random_crop(
            transformed_image, transformed_boxes, transformed_labels, transformed_difficulties)
        transformed_image = FT.to_pil_image(transformed_image)
        if random() < 0.5:
            transformed_image, transformed_boxes = flip(transformed_image, transformed_boxes)

    transformed_image, transformed_boxes = resize(transformed_image, transformed_boxes, size=(300, 300))
    transformed_image = FT.to_tensor(transformed_image)
    transformed_image = FT.normalize(transformed_image, mean=mean, std=std)

    return transformed_image, transformed_boxes, transformed_labels, transformed_difficulties


class PascalVOCDataset(Dataset):
    """
    A dataset class for the Pascal VOC dataset
    """
    def __init__(self, data_folder, data_type, keep_difficult=False):
        """

        Args:
            data_folder (str): a path for the saved json files contain the information of the dataset
            data_type (str): either 'train' for trainig dataset or 'test' for testing dataset
            keep_difficult (bool): True to keep difficult objects
        """
        self.data_folder = data_folder
        self.data_type = data_type
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, f'{self.data_type}_images.json'), 'r') as file:
            self.images = json.load(file)
        with open(os.path.join(data_folder, f'{self.data_type}_objects.json'), 'r') as file:
            self.objects = json.load(file)

    def __getitem__(self, item):
        image = Image.open(self.images[item], 'r').convert('RGB')
        objects = self.objects[item]
        boxes = torch.tensor(objects['boxes'], dtype=torch.float)
        labels = torch.tensor(objects['labels'], dtype=torch.long)
        difficulties = torch.tensor(objects['difficulties'], dtype=torch.bool)

        if not self.keep_difficult:
            boxes = boxes[~difficulties]
            labels = labels[~difficulties]
            difficulties = difficulties[~difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, data_type=self.data_type)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Each image can have different number of objects so need a collate function for the dataloader
        Args:
            batch (iterable): iterable of sets of data from __getitem__ as pass by Dataloader

        Returns:
            (torch.tensor): batch of images as tensor of (N, 3, 300, 300)
            (list): a list of N bounding boxes with N tensor
            (list): a list of labels with N tensor
            (list): a list of N difficulties with N tensor
        """
        images = []
        boxes = []
        labels = []
        difficulties = []

        for item in batch:
            images.append(item[0])
            boxes.append(item[1])
            labels.append(item[2])
            difficulties.append(item[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties
