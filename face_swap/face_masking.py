from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as python_mp
from mediapipe.tasks.python import vision


class FaceMasking:
    def __init__(self):
        landmarks_model_path = Path('models/face_landmarker.task')
        if not landmarks_model_path.exists():
            landmarks_model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            print('Downloading face landmarks model...')
            filename, headers = urlretrieve(url, filename=str(landmarks_model_path))
            print('Download finish!')

        base_options = python_mp.BaseOptions(model_asset_path=str(landmarks_model_path))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_mask(self, image):
        """
        return uint8 mask of the face in image
        Args:
            image (np.ndarray): RGB image with single face

        Returns:
            (np.ndarray): single channel uint8 mask of the face
        """
        im_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.astype(np.uint8).copy())
        detection_result = self.detector.detect(im_mp)

        x = np.array([landmark.x * image.shape[1] for landmark in detection_result.face_landmarks[0]], dtype=np.float32)
        y = np.array([landmark.y * image.shape[0] for landmark in detection_result.face_landmarks[0]], dtype=np.float32)

        hull = np.round(np.squeeze(cv2.convexHull(np.column_stack((x, y))))).astype(np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel)

        return mask