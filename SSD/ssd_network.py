import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def x_y_to_cx_cy(x_y):
    """
    Convert box coordinates from (x_min, y_min, x_max, y_max) to
    (x_center, y_center, width, height)
    Args:
        x_y (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_min, y_min, x_max, y_max) coordinate for each box

    Returns:
        (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each box
    """
    cx = (x_y[:, 0] + x_y[:, 2]) / 2
    cy = (x_y[:, 1] + x_y[:, 3]) / 2
    w = x_y[:, 2] - x_y[:, 0]
    h = x_y[:, 3] - x_y[:, 1]
    return torch.column_stack([cx, cy, w, h])

def cx_cy_to_x_y(cx_cy):
    """
    Convert box coordinates from (x_center, y_center, width, height)
    to (x_min, y_min, x_max, y_max).

    Args:
        cx_cy (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each box.

    Returns:
        (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_min, y_min, x_max, y_max) coordinate for each box.
    """
    x_min = cx_cy[:, 0] - cx_cy[:, 2] / 2
    y_min = cx_cy[:, 1] - cx_cy[:, 3] / 2
    x_max = cx_cy[:, 0] + cx_cy[:, 2] / 2
    y_max = cx_cy[:, 1] + cx_cy[:, 3] / 2
    return torch.column_stack([x_min, y_min, x_max, y_max])

def cx_cy_to_gcx_gcy(cx_cy, prior_cx_cy):
    """
    Convert box coordinates from (x_center, y_center, width, height)
    to an offset coordinate (g_cx, g_cy, g_w, g_h) based on the prior
    coordinates, where:
    g_cx = (c_x - prior_c_x) / prior_w
    g_cy = (c_y - prior_c_y) / prior_h
    g_w = log(w / prior_w)
    g_y = log(h / prior_h)

    Args:
        cx_cy (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each box.
        prior_cx_cy (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each prior box.

    Returns:
        (torch.tensor): A tensor of size (num of boxes, 4) with
            (g_cx, g_cy, g_w, g_h) coordinate for each box.
    """
    gcx = (cx_cy[:, 0] - prior_cx_cy[:, 0]) / (prior_cx_cy[:, 2] / 10)
    gcy = (cx_cy[:, 1] - prior_cx_cy[:, 1]) / (prior_cx_cy[:, 3] / 10)
    gw = torch.log(cx_cy[:, 2] / prior_cx_cy[:, 2]) * 5
    gh = torch.log(cx_cy[:, 3] / prior_cx_cy[:, 3]) * 5
    return torch.column_stack([gcx, gcy, gw, gh])

def gcx_gcy_to_cx_cy(gcx_gcy, prior_cx_cy):
    """
    Convert box coordinates from an offset coordinate (g_cx, g_cy, g_w, g_h)
    to (x_center, y_center, width, height) based on the prior
    coordinates, where:
    g_cx = (c_x - prior_c_x) / prior_w
    g_cy = (c_y - prior_c_y) / prior_h
    g_w = log(w / prior_w)
    g_y = log(h / prior_h)
    Args:
        gcx_gcy (torch.tensor): A tensor of size (num of boxes, 4) with
            (g_cx, g_cy, g_w, g_h) coordinate for each box.
        prior_cx_cy (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each prior box.

    Returns:
        (torch.tensor): A tensor of size (num of boxes, 4) with
            (x_center, y_center, width, height) coordinate for each box.
    """
    cx = gcx_gcy[:, 0] * prior_cx_cy[:, 2] / 10 + prior_cx_cy[:, 0]
    cy = gcx_gcy[:, 1] * prior_cx_cy[:, 3] / 10 + prior_cx_cy[:, 1]
    w = torch.exp(gcx_gcy[:, 2] / 5) * prior_cx_cy[:, 2]
    h = torch.exp(gcx_gcy[:, 3] / 5) * prior_cx_cy[:, 3]
    return torch.column_stack([cx, cy, w, h])

def find_iou(boxes_1, boxes_2=None):
    """
    Get the Intersection over Union (IoU) between each box in boxes_1 with any
    other box in boxes_2. If boxes_2 is None, the IoU is performed over the boxes in
    boxes_1.
    Args:
        boxes_1 (torch.tensor):  A tensor of size (num of boxes, 4) with coordinates
            (x_min, y_min, x_max, y_max).
        boxes_2 (torch.tensor):  A tensor of size (num of boxes, 4) with coordinates
            (x_min, y_min, x_max, y_max).

    Returns (torch.tensor): A tensor of size (num of boxes 1, num of boxes 2) where
        the element in the i-th row and the j-th is the IoU of the i-th box in boxes_1
        and the j-th box in boxes_2

    """
    if boxes_2 is None:
        boxes_2 = boxes_1

    # intersection
    x_min = torch.maximum(boxes_1[:, None, 0], boxes_2[None, :, 0])
    y_min = torch.maximum(boxes_1[:, None, 1], boxes_2[None, :, 1])
    x_max = torch.minimum(boxes_1[:, None, 2], boxes_2[None, :, 2])
    y_max = torch.minimum(boxes_1[:, None, 3], boxes_2[None, :, 3])

    w_intersection = torch.clamp(x_max - x_min, min=0)
    h_intersection = torch.clamp(y_max - y_min, min=0)

    intersection = w_intersection * h_intersection

    w_boxes_1 = boxes_1[:, 2] - boxes_1[:, 0]
    h_boxes_1 = boxes_1[:, 3] - boxes_1[:, 1]
    w_boxes_2 = boxes_2[:, 2] - boxes_2[:, 0]
    h_boxes_2 = boxes_2[:, 3] - boxes_2[:, 1]
    boxes_area_1 = w_boxes_1 * h_boxes_1
    boxes_area_2 = w_boxes_2 * h_boxes_2

    union = boxes_area_1[:, None] + boxes_area_2[None, :] - intersection

    return intersection / union


class ModifiedVGG16(nn.Module):
    """
    A VGG16 model use as a backbone for the SSD with the following modification
    1. Change max pooling for 3-th block to use ceiling  instead of floor to get even dimensions
    2. Change the max pooling for the 5-th block
    3. Replace the fully connected layers (6 and 7) with convolution layers decimated based on
        the original layers.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=False)
        features = list(vgg.features)
        features[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # change max pooling for 3-th block
        features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # change max pooling for 5th block
    
        self.features_1 = nn.Sequential(*features[:23])  # The part of the model that output the features map of conv4_3

        conv_6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv_7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # get the weights of the fully connected layers 6 and 7 and change the view to a convolution like order
        fc6_weights = vgg.classifier[0].state_dict()['weight'].view(4096, 512, 7, 7)
        fc6_bias = vgg.classifier[0].state_dict()['bias']
        fc7_weights = vgg.classifier[3].state_dict()['weight'].view(4096, 4096, 1, 1)
        fc7_bias = vgg.classifier[3].state_dict()['bias']

        # decimation for fc6 from (4096, 512, 7, 7) to (1024, 512, 3, 3)
        fc6_weights = torch.index_select(fc6_weights, 0, torch.arange(0, fc6_weights.size(0), 4, dtype=torch.long))
        fc6_weights = torch.index_select(fc6_weights, 2, torch.arange(0, fc6_weights.size(2), 3, dtype=torch.long))
        fc6_weights = torch.index_select(fc6_weights, 3, torch.arange(0, fc6_weights.size(3), 3, dtype=torch.long))
        fc6_bias = torch.index_select(fc6_bias, 0, torch.arange(0, fc6_bias.size(0), 4, dtype=torch.long))
        # decimation for fc7 from (4096, 4096, 1, 1) to (1024, 1024, 1, 1)
        fc7_weights = torch.index_select(fc7_weights, 0, torch.arange(0, fc6_weights.size(0), 4, dtype=torch.long))
        fc7_weights = torch.index_select(fc7_weights, 1, torch.arange(0, fc6_weights.size(1), 4, dtype=torch.long))
        fc7_bias = torch.index_select(fc7_bias, 0, torch.arange(0, fc7_bias.size(0), 4, dtype=torch.long))

        # load the decimated weights to the new conv layers
        conv_6.state_dict()['weight'] = fc6_weights
        conv_6.state_dict()['bias'] = fc6_bias
        conv_7.state_dict()['weight'] = fc7_weights
        conv_7.state_dict()['bias'] = fc7_bias

        # The second part of the model that get the features map from conv4_3
        # and its output is the features map of conv7
        self.features_2 = nn.Sequential(
            *(features[23:] +  [
                conv_6,
                nn.ReLU(inplace=True),
                conv_7,
                nn.ReLU(inplace=True)
            ])
        )

    def forward(self, images):
        """
        Forward propagation.
        Args:
            images (torch.tensor): a batch of images as tensors of (N, 3, 300, 300)

        Returns:
            (torch.tensor): the features map of conv4_3 (N, 512, 38, 38)
                and conv7 (N, 1024, 19, 19) that will be use for the SSD
        """
        conv4_3_features = self.features_1(images)
        conv7_features = self.features_2(conv4_3_features)

        return conv4_3_features, conv7_features


class AdditionalConvLayers(nn.Module):
    """
    Additional convolution layers of the bigger objects
    """
    def __init__(self):
        super().__init__()
        self.conv8 = self.make_block(1024, 256, 512, stride=2, padding=1)
        self.conv9 = self.make_block(512, 128, 256, stride=2, padding=1)
        self.conv10 = self.make_block(256, 128, 256)
        self.conv11 = self.make_block(256, 128, 256)

    def make_block(self, input, middle, output, stride=1, padding=0):
        """
        A routine to create and initalize the weights of a block of convolution layers
        Args:
            input (int): number of channels of the input tensor
            middle (int): number of channels of the input of the second convolution layer
            output (int): number of output channels of the block
            stride (int): the stride of the second convolution layer in the block
            padding (int): the padding for the second layer in the block

        Returns:
            (torch.nn.Sequential): a block of convolution layers with ReLU initialized by xavier_uniform
        """
        block = nn.Sequential(
            nn.Conv2d(input, middle, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, output, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

        # initialize
        nn.init.xavier_uniform_(block[0].weight)
        nn.init.constant_(block[0].bias, 0.0)
        nn.init.xavier_uniform_(block[2].weight)
        nn.init.constant_(block[2].bias, 0.0)

        return block

    def forward(self, conv7_features):
        """
        Forward propagation
        Args:
            conv7_features (torch.tensor): features map from conv7 (of ModifiedVGG16) of size (N, 1024, 19, 19)

        Returns:
            (torch.tensor): features map of layer conv8_2 of size (N, 512, 10, 10)
            (torch.tensor): features map of layer conv9_2 of size (N, 256, 5, 5)
            (torch.tensor): features map of layer conv10_2 of size (N, 256, 3, 3)
            (torch.tensor): features map of layer conv11_2 of size (N, 256, 1, 1)
        """
        conv8_2_features = self.conv8(conv7_features)
        conv9_2_features = self.conv9(conv8_2_features)
        conv10_2_features = self.conv10(conv9_2_features)
        conv11_2_features = self.conv11(conv10_2_features)

        return conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features


class PredictionConv(nn.Module):
    """
    The part of the network that use the features maps to predict the location and class
    of objects in the image, based on features maps from conv4_3, conv7 conv8_2, conv9_2,
    conv10_2 and conv11_2
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # localization convolutions. The output channels number is the number
        # of prior boxes multiply by 4 coordinate of a box
        self.loc_conv4_3 = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, 4 * 6, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, 4 * 6, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, 4 * 6, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)

        # class prediction convolution. The output channels number is the number
        # of prior boxes multiply by the number of classes
        self.class_conv4_3 = nn.Conv2d(512, self.num_classes * 4, kernel_size=3, padding=1)
        self.class_conv7 = nn.Conv2d(1024, self.num_classes * 6, kernel_size=3, padding=1)
        self.class_conv8_2 = nn.Conv2d(512, self.num_classes * 6, kernel_size=3, padding=1)
        self.class_conv9_2 = nn.Conv2d(256, self.num_classes * 6, kernel_size=3, padding=1)
        self.class_conv10_2 = nn.Conv2d(256, self.num_classes * 4, kernel_size=3, padding=1)
        self.class_conv11_2 = nn.Conv2d(256, self.num_classes * 4, kernel_size=3, padding=1)

        # initialize the convolution layers
        for conv in self.children():
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

    def arrange_prediction(self, layer, features, last_dim_size):
        """
        A function to rearrange the size of the output from a layer.
        Args:
            layer (torch.nn.Model): a layer of network
            features (torch.tensor): features map
            last_dim_size (int): the size of the last dimension of the output

        Returns:
            (torch.tensor): rearrange form of a features map results from layer(features)
        """
        batch_size = features.shape[0]
        result = layer(features)  # (N, num_of_boxes * last_dim, layer_h, layer_w)
        result = result.permute(0, 2, 3, 1).contiguous()  # (N, layer_h, layer_w, num_of_boxes * last_dim)
        result = result.view(batch_size, -1, last_dim_size)  # (N, number of boxes, last_dim)
        return result

    def forward(self, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features,
                conv11_2_features):
        """
        Forward propagation
        Args:
            conv4_3_features (torch.tensor): the features maps from layer conv4_3 of size (N, 512, 38, 38)
            conv7_features (torch.tensor): the features maps from layer conv7 of size (N, 1024, 19, 19)
            conv8_2_features (torch.tensor): the features maps from layer conv8_2 of size (N, 512, 10, 10)
            conv9_2_features (torch.tensor): the features maps from layer conv9_2 of size (N, 256, 5, 5)
            conv10_2_features (torch.tensor): the features maps from layer conv10_2 of size (N, 256, 3, 3)
            conv11_2_features (torch.tensor): the features maps from layer conv11_2 (N, 256, 1, 1)

        Returns:
            (torch.tensor): predication of the location of object's box, size (N, 8732, 4)
            (torch.tensor): prediction of the object's class using scores for each possible
                class, size (N, 8732, n_classes)
        """

        loc_conv4_3_prediction = self.arrange_prediction(self.loc_conv4_3, conv4_3_features, 4)
        loc_conv7_prediction = self.arrange_prediction(self.loc_conv7, conv7_features, 4)
        loc_conv8_2_prediction = self.arrange_prediction(self.loc_conv8_2, conv8_2_features, 4)
        loc_conv9_2_prediction = self.arrange_prediction(self.loc_conv9_2, conv9_2_features, 4)
        loc_conv10_2_prediction = self.arrange_prediction(self.loc_conv10_2, conv10_2_features, 4)
        loc_conv11_2_prediction = self.arrange_prediction(self.loc_conv11_2, conv11_2_features, 4)

        class_conv4_3_prediction = self.arrange_prediction(self.class_conv4_3, conv4_3_features, self.num_classes)
        class_conv7_prediction = self.arrange_prediction(self.class_conv7, conv7_features, self.num_classes)
        class_conv8_2_prediction = self.arrange_prediction(self.class_conv8_2, conv8_2_features, self.num_classes)
        class_conv9_2_prediction = self.arrange_prediction(self.class_conv9_2, conv9_2_features, self.num_classes)
        class_conv10_2_prediction = self.arrange_prediction(self.class_conv10_2, conv10_2_features, self.num_classes)
        class_conv11_2_prediction = self.arrange_prediction(self.class_conv11_2, conv11_2_features, self.num_classes)

        locations = torch.cat([loc_conv4_3_prediction, loc_conv7_prediction, loc_conv8_2_prediction,
                              loc_conv9_2_prediction, loc_conv10_2_prediction, loc_conv11_2_prediction],
                              dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([class_conv4_3_prediction, class_conv7_prediction, class_conv8_2_prediction,
                               class_conv9_2_prediction, class_conv10_2_prediction, class_conv11_2_prediction],
                              dim=1)  # (N, 8732, num of classes)
        return locations, classes_scores


class SSD(nn.Module):
    """
    SSD300 composed of the ModifiedVGG, AdditionalConvLayers and PredictionConv
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ModifiedVGG16()
        self.additional_convs = AdditionalConvLayers()
        self.prediction_convs = PredictionConv(num_classes)

        # Since the scale of the features map of conv4_3 is larger than the other maps we need to rescale it.
        # We use L2 normalization with initial factor of 20, while its factor will be learned during the training
        self.rescale_factor = nn.Parameter(torch.full((1, 512, 1, 1), 20., dtype=torch.float))

        self.prior_cx_cy = self.create_prior_boxes()

    def create_prior_boxes(self):
        """
        Create the 8732 default prior boxes. The number of prior boxes for the features maps is as following:
        conv4_3 - 4 boxes for 38x38 locations = 5776
        conv7 - 6 boxes for 19x19 locations = 2166
        conv8_2 - 6 boxes for 10x10 locations = 600
        conv9_2 - 6 boxes for 5x5 locations = 150
        conv10_2 - 4 boxes for 3x3 locations = 36
        conv11_2 - 4 boxes for 1x1 locations = 4

        The width and height of the prior boxes are defined using the scale (s) and aspect ratio (a):
        w * h = s ** 2
        w / h = a
        From here:
        w = s * sqrt(a)
        h = s / sqrt(a)
        Returns:
            (torch.tensor): prior boxes with coordinates (x_center, y_center, width, height) of size (8732, 4)
        """
        features_map_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        aspect_ratio = {'conv4_3': [1., 2., 0.5], 'conv7': [1., 2., 3., 0.5, .333], 'conv8_2': [1., 2., 3., 0.5, .333],
                        'conv9_2': [1., 2., 3., 0.5, .333], 'conv10_2': [1., 2., 0.5], 'conv11_2': [1., 2., 0.5]}

        prior_boxes = []

        for k, (features_map, dim) in enumerate(features_map_dims.items()):
            for i in range(dim):
                cy = (i + 0.5) / dim
                for j in range(dim):
                    cx = (j + 0.5) / dim

                    for ratio in aspect_ratio[features_map]:
                        prior_boxes.append([cx, cy, scales[features_map] * np.sqrt(ratio),
                                            scales[features_map] / np.sqrt(ratio)])
                        # For aspect ratio of 1, an additional box is defined with a scale equal to the
                        # geometric mean of the current features map scale and the next one
                        if ratio == 1:
                            if k < len(features_map_dims) - 1:
                                additional_scale = np.sqrt(scales[features_map] * scales[list(features_map_dims)[k+1]])
                            else:  # there is no next features map
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.tensor(prior_boxes, dtype=torch.float, device=device)
        prior_boxes_x_y = cx_cy_to_x_y(prior_boxes)
        prior_boxes_x_y.clamp_(0, 1)  # clamp the box if it outside the image
        prior_boxes = x_y_to_cx_cy(prior_boxes_x_y)

        return prior_boxes

    def forward(self, images):
        """
        Forward propagation
        Args:
            images (torch.tensor): batch of images of size (N, 3, 300, 300)

        Returns:
            (torch.tensor): the predicted locations of the boxes of the objects (N, 8732, 4)
            (torch.tensor): the predicted classes scores of the objects (N, 8732, num of classes)
        """
        # get the features maps from the backbone
        conv4_3_features, conv7_features = self.backbone(images)

        # rescale conv4_3 features
        norm_conv4_3 = torch.sqrt(torch.sum(conv4_3_features ** 2, dim=1, keepdim=True))
        conv4_3_features_normalized = conv4_3_features / norm_conv4_3
        conv4_3_features = conv4_3_features_normalized * self.rescale_factor

        conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features = self.additional_convs(conv7_features)
        locations, classes_scores = self.prediction_convs(conv4_3_features, conv7_features, conv8_2_features,
                                                          conv9_2_features, conv10_2_features, conv11_2_features)

        return locations, classes_scores

    def detect_objects(self, predicted_locations, predicted_classes, min_score, max_overlap, top_k):
        """
        Use the results of the predicted locations and classes scores, to make an object prediction.
        Find boxes with scores above the minimum threshold and perform non-maximum suppression.
        Args:
            predicted_locations (torch.tensor): the predicted locations of objects boxes (N, 8732, 4)
            predicted_classes (torch.tensor): the predicted scores for each class (N, 8732, num of classes)
            min_score (float): the minimum score a box needs to considered as detection of a class
            max_overlap (float): the maximum overlap two boxes can have and still considered as two separate
                detection. If the overlap is greater than this value, the box with the lower score will be suppressed.
            top_k (int): keep only the k boxes with the highest scores

        Returns:
            (list): the boxes of detection
            (list): the labels of detection
            (list): the scores of detection
        """
        batch_size = predicted_locations.shape[0]
        # convert scores to probabilities using soft-max
        predicted_classes = F.softmax(predicted_classes, dim=2)

        # initialize empty list to store the final results of detections
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for ii in range(batch_size):
            # convert the locations coordinates to (x_min, y_min, x_max, y_max)
            decoded_locations = cx_cy_to_x_y(gcx_gcy_to_cx_cy(predicted_locations[ii], self.prior_cx_cy))
            # initialize empty list to store the current image results of detections
            image_boxes = []
            image_labels = []
            image_scores = []

            # go over each class
            for c in range(1, self.num_classes):
                class_score = predicted_classes[ii][:, c]  # get the class score for the current image for each box
                scores_above_min_score = class_score > min_score
                num_of_scores_above_min_score = torch.sum(scores_above_min_score).item()
                # if there is no any box with score above the minimum score, there is no object and can continue
                # to the next class
                if num_of_scores_above_min_score == 0:
                    continue

                # keep only boxes with score higher than min_score
                high_class_scores = class_score[scores_above_min_score]
                high_class_decoded_locations = decoded_locations[scores_above_min_score]

                # sort the boxes by score from the higher to lower
                high_class_scores_sorted, sort_indexes = torch.sort(high_class_scores, dim=0, descending=True)
                high_class_decoded_locations_sorted = high_class_decoded_locations[sort_indexes]

                # get the overlap between the boxes, so we can suppress boxes with high overlap
                overlap = find_iou(high_class_decoded_locations_sorted)
                boxes_to_suppress = torch.zeros(num_of_scores_above_min_score, dtype=torch.bool, device=device)

                # go over the boxes
                for ii_box in range(num_of_scores_above_min_score):
                    # if the current boxes already marked to suppressed, continue
                    if boxes_to_suppress[ii_box]:
                        continue
                    # mark boxes with overlap greater than max_overlap as boxes to suppress
                    boxes_to_suppress = torch.maximum(boxes_to_suppress, overlap[ii_box] > max_overlap)
                    boxes_to_suppress[ii_box] = 0  # don't suppress the current box (it has overlap of 1 with itself)

                # keep boxes which has not been marked to suppress
                image_boxes.append(high_class_decoded_locations_sorted[~boxes_to_suppress])
                image_labels.append(torch.full((torch.sum(~boxes_to_suppress).item(), ), c, dtype=torch.long, device=device))
                image_scores.append(high_class_scores_sorted[~boxes_to_suppress])

            # if no boxes found that meet the criterion, add default values
            if not image_boxes:
                image_boxes.append(torch.tensor([0, 0, 1, 1], dtype=torch.float, device=device))
                image_labels.append(torch.tensor([0], dtype=torch.long, device=device))
                image_scores.append(torch.tensor([0], dtype=torch.float, device=device))

            # concatenate the result for the current image
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            num_of_founded_objects = image_scores.shape[0]

            # keep at maximum the top k boxes
            if num_of_founded_objects > top_k:
                image_scores, top_k_indexes = torch.topk(image_scores, top_k)
                image_boxes = image_boxes[top_k_indexes, :]
                image_labels = image_labels[top_k_indexes]

            # add to the final results of detections
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class SSDLoss(nn.Module):
    """
    The module to implement the loss function for the SSD
    """
    def __init__(self, priors_cx_cy, threshold=0.5, neg_pos_ratio=3, alpha=1):
        """
        Args:
            priors_cx_cy (torch.tensor): prior boxes with coordinates (x_center, y_center, width, height) of size (8732, 4)
            threshold (float): minimum overlap a prior should have with ground truth box to not considered as background
            neg_pos_ratio (int): the ration between the number of negative detections to the positive detections found
            alpha (float): scale factor for the class classification loss in the total loss calculation
        """
        super().__init__()
        self.prior_cx_cy = priors_cx_cy
        self.prior_x_y = cx_cy_to_x_y(priors_cx_cy)
        self.threshold = threshold
        self.neg_pos_ration = neg_pos_ratio
        self.alpha = alpha

        self.localization_loss = nn.L1Loss()
        self.class_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locations, predicted_class_scores, gt_boxes, gt_labels):
        """
        Forward propagation which calculate the loss
        Args:
            predicted_locations (torch.tensor): the predicted locations of objects boxes (N, 8732, 4)
            predicted_class_scores (torch.tensor): the predicted scores for each class (N, 8732, num of classes)
            gt_boxes (list[torch.tensor]): ground truth bounding box for objects with coordinates
                (x_min, y_min, x_max, y_max)
            gt_labels (list[torch.tensor]): ground truth labels for objects

        Returns:
            (torch.tensor): tensor with the loss value as scalar
        """
        batch_size = predicted_locations.shape[0]
        num_of_priors = self.prior_cx_cy.shape[0]
        num_of_classes = predicted_class_scores.shape[2]

        true_locations = torch.zeros((batch_size, num_of_priors, 4), dtype=torch.float, device=device)
        true_classes = torch.zeros((batch_size, num_of_priors), dtype=torch.long, device=device)

        # go over each image in the batch
        for ii in range(batch_size):
            num_of_object_in_image = gt_boxes[ii].shape[0]
            # get the overlap between the ground truth boxes and the prior boxes
            overlap = find_iou(gt_boxes[ii], self.prior_x_y)  # (num objects, 8732)

            # for each prior box, find the ground truth box with the maximum overlap
            max_overlap_per_prior, object_per_prior = torch.max(overlap, dim=0)  # (8732)
            # for each object, find the prior box with the maximum overlap with its box
            _, prior_per_object = torch.max(overlap, dim=1)  # (num objects)
            # to prevent a situation where an object is not assigned to any prior, we use the prior_pre_object
            # to ensure each object is assigned to at least one prior
            object_per_prior[prior_per_object] = torch.arange(num_of_object_in_image, dtype=torch.long, device=device)
            # to ensure the previous step won't be considered as background (and so again we will have object without
            # prior), we artificially give these priors an overlap greater than the threshold
            max_overlap_per_prior[prior_per_object] = 1
            print(gt_labels[ii][object_per_prior][:5])
            labels_per_prior = gt_labels[ii][object_per_prior]  # get label for each prior box (8732)
            # if overlap with ground truth box is less than the threshold, set as background
            labels_per_prior[max_overlap_per_prior < self.threshold] = 0
            # store the ground truth labels and boxes (converted to offset form) for each prior
            true_classes[ii] = labels_per_prior
            true_locations[ii] = cx_cy_to_gcx_gcy(x_y_to_cx_cy(gt_boxes[ii][object_per_prior]), self.prior_cx_cy)

        positive_priors = true_classes != 0  # prior with object (non-background), (N, 8732)
        #  compute the localization loss using only non-background priors
        localization_loss = self.localization_loss(predicted_locations[positive_priors], true_locations[positive_priors])

        num_of_positive_priors = torch.sum(positive_priors, dim=1)
        #  number of negative detection to consider. These will be the hardest negative
        num_of_hard_negative_priors = self.neg_pos_ration * num_of_positive_priors

        # compute loss for classification for all priors
        class_loss_all = self.class_loss(predicted_class_scores.view(-1, num_of_classes), true_classes.view(-1))
        class_loss_all = class_loss_all.view(batch_size, num_of_priors)

        # get only loss for positive detection
        class_loss_positive = class_loss_all[positive_priors]

        # get loss for the hardest negative detections (these with the highest loss)
        class_loss_negative = class_loss_all.clone()  # (N, 8732)
        class_loss_negative[positive_priors] = 0  # set the loss of the positive detection to 0
        class_loss_negative, _ = torch.sort(class_loss_negative, dim=1, descending=True)
        hardness_rank = torch.arange(num_of_priors, dtype=torch.long).unsqueeze(0).expand_as(class_loss_negative).to(device)
        hard_negative = hardness_rank < num_of_hard_negative_priors.unsqueeze(1)
        class_loss_hard_negative = class_loss_negative[hard_negative]

        # compute overall loss for classification, and normalized over the positive prior only
        class_loss = 1 / torch.sum(num_of_positive_priors) * (torch.sum(class_loss_positive) + torch.sum(class_loss_hard_negative))

        loss = class_loss + self.alpha * localization_loss  # total loss
        return loss





