import warnings
from shutil import copy, rmtree
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import matplotlib.pyplot as plt


class YoloWrapper:
    def __init__(self, model_weights: str) -> None:
        """
        Initialize YOLOv8 model with weights.
        Args:
            model_weights (str): model weight can be one of the follows:
                - 'nano' for YOLOv8 nano model
                - 'small' for YOLOv8 small model
                - a path to a .pt file contains the weights from a previous training.
        """
        if model_weights == 'nano':
            model_weights = 'yolov8n.pt'
        elif model_weights == 'small':
            model_weights = 'yolov8s.pt'
        elif model_weights == 'medium':
            model_weights = 'yolov8m.pt'
        elif (not Path(model_weights).exists()) or (Path(model_weights).suffix != '.pt'):
            raise ValueError('The parameter model_weight should be "nano", "small" or a'
                             'path to a .pt file with saved weights')

        # initialize YOLO model
        self.model = YOLO(model_weights)

    def train(self, config: str, epochs: int = 100, name: str = None) -> None:
        """
        Train the model. After running a 'runs/detect/<name>' folder will be created and stores information
        about the training and the saved weights.
        Args:
            config (str): a path to a configuration yaml file for training.
                Such a file contains:
                    path -  absolute path to the folder contains the images and labels folders with the data
                    train - relative path to 'path' of the train images folder (images/train)
                    val -  relative path to 'path' of the validation images folder (images/val), if exists
                    nc - the number of classes
                    names - a list of the classes names
                Can be created with the create_config_file method.
            epochs (int): number of epochs for training
            name (str): the name of the results' folder. If None (default) a default name 'train #' will
                be created.

        Returns:

        """
        if Path(config).suffix != '.yaml':
            raise ValueError('Config file should be a yaml file')
        self.model.train(data=config, epochs=epochs, name=name, freeze=10)

    def predict(self, image: str | Path | np.ndarray | list[str] | list[Path] | list[np.ndarray], threshold: float = 0.25,
                ) -> list[np.ndarray]:
        """
        Predict bounding box for images.
        Args:
            image (str|Path|np.ndarray|list[str]|list[Path]|list[np.ndarray]): image data. Can be a string path
                to an image, a BGR image as numpy ndarray, a list with string paths to images or a list
                with BGR images as numpy ndarray.
            threshold (float): a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.

        Returns:
            (list[np.ndarray]): a list with numpy ndarrays for each detection
        """
        yolo_results = self.model(image, conf=threshold)
        bounding_boxes = [torch.concatenate([x.boxes.xyxy[:, :2], x.boxes.xyxy[:, 2:] - x.boxes.xyxy[:, :2]], dim=1).cpu().numpy()
                          for x in yolo_results]
        return bounding_boxes

    def predict_and_save_to_csv(self, images: list[str] | list[Path] | list[np.ndarray], image_ids: list[str] = None,
                                path_to_save_csv: str | Path = '', threshold: float = 0.25, minimum_size: int = 100,
                                only_most_conf=True) -> None:
        """
        Predict a batch of images and return the bounding boxs prediction in a csv file with the columns:
        image_id, x_top_left, y_top_left, width, height. If there is no any prediction, a csv will not be created.
        Args:
            images (list[str] | list[Path] | list[np.ndarray]): a list with string paths to images or a list
                with BGR images as numpy ndarray.
            image_ids (list[str]): the ids of the images
            path_to_save_csv (Optional, str|Path): a path where to save the csv file. If the path is not for
                a specific csv file, the file name will be bounding_box.csv by default.
            threshold (float):  a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.
            minimum_size (int): the minimum width and height in pixels for a bounding box to saved in the csv file.
                If the bounding box founded is smaller than the minimum size, the width and the height will update
                 to the minimum size, and the top left point will be updated accordingly to keep the object in
                 the center
            only_most_conf (bool): True to keep only the bounding box with the highest confidence for each image.
                The bounding boxes are sorted so the first one is the one with the highest confidence

        Returns:

        """
        if image_ids is None:
            if isinstance(images[0], np.ndarray):
                raise ValueError('image_ids can not be None if images is a list of numpy arrays')
            else:
                # get the name of the images as image id
                image_ids = [Path(image_path).stem for image_path in images]

        if isinstance(images[0], (str, Path)):
            h, w, _ = cv2.imread(str(images[0])).shape
        else:  # numpy array
            h, w, _ = images[0].shape

        bbox_list = self.predict(images, threshold)
        if only_most_conf:  # keep only the bounding box with the highest confidence
            bbox_list = [bboxes[[0], :] if bboxes.shape[0] > 0 else bboxes for bboxes in bbox_list]

        # if there are more than one bounding box for an image, we need to duplicate the image id
        image_ids_with_duplicates = [image_id
                                     for bbox, image_id in zip(bbox_list, image_ids)
                                     for _ in range(bbox.shape[0])]
        bbox_matrix = np.vstack(bbox_list)

        if bbox_matrix.shape[0] == 0:
            warnings.warn('A bounding boxes were not found for any of the images.'
                          'A csv file will not be created')
            return

        # set the width to minimum value
        less_than_min = bbox_matrix[:, 2] < minimum_size
        missing_width = minimum_size - bbox_matrix[less_than_min, 2]
        bbox_matrix[less_than_min, 2] = minimum_size
        bbox_matrix[less_than_min, 0] = np.minimum(
            np.maximum(bbox_matrix[less_than_min, 0] - missing_width / 2, 0),
            w - 1 - minimum_size
        )

        # set the height to minimum value
        less_than_min = bbox_matrix[:, 3] < minimum_size
        missing_height = minimum_size - bbox_matrix[less_than_min, 3]
        bbox_matrix[less_than_min, 3] = minimum_size
        bbox_matrix[less_than_min, 1] = np.minimum(
            np.maximum(bbox_matrix[less_than_min, 1] - missing_height / 2, 0),
            h - 1 - minimum_size
        )

        dict_for_csv = {
            'image_id': image_ids_with_duplicates,
            'x_top_left': bbox_matrix[:, 0],
            'y_top_left': bbox_matrix[:, 1],
            'width': bbox_matrix[:, 2],
            'height': bbox_matrix[:, 3]
        }

        bbox_dataframe = pd.DataFrame(dict_for_csv)

        path_to_save_csv = Path(path_to_save_csv)
        if path_to_save_csv.suffix == '':
            path_to_save_csv = path_to_save_csv / 'bounding_boxes.csv'
        if path_to_save_csv.suffix != '.csv':
            raise ValueError('A non-csv file is given')
        path_to_save_csv.parent.mkdir(parents=True, exist_ok=True)
        bbox_dataframe.to_csv(str(path_to_save_csv), index=False)

    def predict_and_show(self, image: str | np.ndarray, threshold: float = 0.25) -> None:
        """
        Predict bounding box for a single image and show the bounding box with its confidence.
        Args:
            image (str | np.ndarray): a path to an image or a BGR np.ndarray image to predict
                bounding box for
            threshold (float): a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.
        Returns:

        """
        yolo_results = self.model(image, threshold=threshold)
        labeled_image = yolo_results[0].plot()
        plt.figure()
        plt.imshow(labeled_image[..., ::-1])
        plt.show()

    @staticmethod
    def draw_bbox_from_csv(image: str | Path | np.ndarray, csv_path: str, image_id: str = None) -> None:
        """
        Draw bounding boxes according to a csv for one image
        Args:
            image (str | Path | np.ndarray): a path to a single image or a BGR np.ndarray image to predict
                bounding box for
            csv_path (str): a path to a csv file with bounding boxes
            image_id (str): the image id as in the csv. If None (default), all the bounding boxes
                in the csv will be drawn.
        Returns:

        """
        if not isinstance(image, np.ndarray):
            image = cv2.imread(str(image))

        bbox_dataframe = pd.read_csv(csv_path)

        if image_id is not None:
            bbox_dataframe = bbox_dataframe.loc[bbox_dataframe['image_id'] == image_id, :]

        for _, row in bbox_dataframe.iterrows():
            cv2.rectangle(
                image,
                (int(row['x_top_left']), int(row['y_top_left'])), (int(row['x_top_left'] + row['width']), int(row['y_top_left'] + row['height'])),
                (0, 0, 255),
                4
            )

        plt.figure()
        plt.imshow(image[..., ::-1])
        if image_id is not None:
            plt.title(image_id)
        plt.show()

    @staticmethod
    def create_config_file(parent_data_path: str | Path, class_names: list[str], path_to_save: str = None) -> None:
        """
        Create YOLOv8 configuration yaml file. The configuration file contains:
        path -  absolute path to the folder contains the images and labels folders with the data
        train - relative path to 'path' of the train images folder (images/train)
        val -  relative path to 'path' of the validation images folder (images/val), if exists
        nc - the number of classes
        names - a list of the classes names
        Args:
            parent_data_path (str|Path): path to the folder contains the images and labels folder with the data.
                The structure of this folder should be:
                - parent_data_path
                    - images
                        - train
                        - val (optional)
                    - labels
                        - train
                        - val (optional)
            class_names (list[str]): a list contains the names of the classes. The first name is for label 0, and so on
            path_to_save (Optional, str): A path to where to save the result. By defulat it save it in the working
                directory as 'config.yaml'. If a folder is given a file 'config.yaml' will be saved inside. If a path
                including file name is given, the file must be with a .yaml suffix.

        Returns:

        """
        parent_data_path = Path(parent_data_path)
        if not parent_data_path.exists():
            raise FileNotFoundError(f'Folder {parent_data_path} is not found')
        if not (parent_data_path / 'images' / 'train').exists():
            raise FileNotFoundError(f'There is not folder {parent_data_path / "images" / "train"}')
        if not (parent_data_path / 'labels' / 'train').exists():
            raise FileNotFoundError(f'There is not folder {parent_data_path / "labels" / "train"}')

        config = {
            'path': str(parent_data_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }

        if not (parent_data_path / 'images' / 'val').exists():
            config.pop('val')

        if path_to_save is None:
            path_to_save = 'config.yaml'
        path_to_save = Path(path_to_save)

        if not path_to_save.suffix:  # is a folder
            path_to_save.mkdir(parents=True, exist_ok=True)
            path_to_save = path_to_save / 'config.yaml'

        if path_to_save.suffix != '.yaml':
            raise ValueError(f'The path to save the configuration file should be a folder, a yaml file or None.'
                             f'Got a {path_to_save.suffix} file instead')

        with open(path_to_save, 'w') as file:
            for key, value in config.items():
                file.write(f'{key}: {value}\n')

    @staticmethod
    def create_yolo_labels_from_crop(images_path: str | Path, crops_path: str | Path,
                                     labels_path: str | Path | None = None) -> None:
        """
        Create labels in YOLO format from images cropped from larger images.
        The YOLO format is a txt file where there is a row for each object
        at the format: <class:int> <x center: float> <y center: float> <width: float> <height: float>
        where all measure in a relative coordinate in the range [0,  1].
        The function assume the folder of the original and cropped images have the same data and the images
        have the same name.
        Args:
            images_path (str|Path): path to the folder containing the full images
            crops_path (str|Path): path to the folder containing the cropped images
            labels_path (str|Path|None): optional (default None). Path to the folder
                where the train will be saved. If None the labels will be saved in a
                labels folder in the parent directory of the images.

        Returns:

        """
        images_path = Path(images_path)
        crops_path = Path(crops_path)

        if labels_path is None:
            parent_dir = images_path.parent
            labels_path = parent_dir / 'labels'
        else:
            labels_path = Path(labels_path)

        labels_path.mkdir(parents=True, exist_ok=True)

        images_list = sorted(list(images_path.glob('*')))
        crops_list = sorted(list(crops_path.glob('*')))

        for image_path, crop_path in zip(images_list, crops_list):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)

            # Apply template Matching
            template_match_results = cv2.matchTemplate(image, crop, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match_results)
            x_left = max_loc[0]
            y_top = max_loc[1]
            h, w = crop.shape
            x_right = x_left + w
            y_bottom = y_top + h

            with open((labels_path / image_path.stem).with_suffix('.txt'), 'w') as file:
                text = (f'0 '
                        f'{((x_left + x_right) / 2) / image.shape[1]} '
                        f'{((y_top + y_bottom) / 2) / image.shape[0]} '
                        f'{w / image.shape[1]} '
                        f'{h / image.shape[0]}')
                file.write(text)

    @staticmethod
    def create_dataset(images_path: str | Path, labels_path: str | Path = None, result_path: str | Path = None,
                       train_size: float = 0.9) -> None:
        """
        Create A YOLO dataset from a folder of images and a folder of labels. The function
        assumes all the images have a labels with the same name. The output structure is
        - result_path
            - images
                - train
                - val (optional)
            - labels
                - train
                - val (optional)
        Args:
            images_path (str|Path): path to the folder contains the images
            labels_path (str|Path): path to the folder contains the labels
            result_path (optional, str|Path): path to the folder where the result will be saved.
                If it's None, a folder named 'data' will be created in parent directory of the images.
            train_size (float): a number between 0 and 1 represent the proportion of the dataset to
                include in the train split

        Returns:

        """
        if train_size <= 0 or 1 < train_size:
            raise ValueError(f'Train size should be between 0 to 1, but got {train_size}')

        images_path = Path(images_path)
        labels_path = Path(labels_path)

        if result_path is None:
            parent_dir = images_path.parent
            result_path = parent_dir / 'data'
        else:
            result_path = Path(result_path)

        if result_path.exists():
            rmtree(result_path)

        all_images = sorted(list(images_path.glob('*')))
        all_labels = sorted(list(labels_path.glob('*')))

        training_dataset, val_dataset, train_labels, val_labels = train_test_split(
            all_images, all_labels, train_size=train_size)

        result_path_image_training = result_path / 'images' / 'train'
        result_path_image_training.mkdir(parents=True, exist_ok=False)
        result_path_label_training = result_path / 'labels' / 'train'
        result_path_label_training.mkdir(parents=True, exist_ok=False)

        for image, label in zip(training_dataset, train_labels):
            copy(image, result_path_image_training / image.name)
            copy(label, result_path_label_training / label.name)

        if val_dataset:
            result_path_image_validation = result_path / 'images' / 'val'
            result_path_image_validation.mkdir(parents=True, exist_ok=False)
            result_path_label_validation = result_path / 'labels' / 'val'
            result_path_label_validation.mkdir(parents=True, exist_ok=False)

            for image, label in zip(val_dataset, val_labels):
                copy(image, result_path_image_validation / image.name)
                copy(label, result_path_label_validation / label.name)