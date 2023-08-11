import cv2
from pathlib import Path
from shutil import rmtree
from urllib.request import urlretrieve
import numpy as np
from tqdm import tqdm
from typing import Union

def extract_frames_from_video(video_path: Union[str, Path], output_folder: Union[str, Path], frames_to_skip: int=0) -> None:
    """
    Extract frame from video as a JPG images.
    Args:
        video_path (str|Path): the path to the input video from it the frame will be extracted
        output_folder (str|Path): the folder where the frames will be saved
        frames_to_skip (int): how many frames to skip after a frame which is saved. 0 will save all the frames.
            If, for example, this value is 2, the first frame will be saved, then frame 2 and 3 will be skipped,
            the 4th frame will be saved, and so on.

    Returns:

    """

    video_path = Path(video_path)
    output_folder = Path(output_folder)

    if not video_path.exists():
        raise ValueError(f'The path to the video file {video_path.absolute()} is not exist')
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    video_capture = cv2.VideoCapture(str(video_path))

    extract_frame_counter = 0
    saved_frame_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if extract_frame_counter % (frames_to_skip + 1) == 0:
            cv2.imwrite(str(output_folder / f'{saved_frame_counter:05d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_frame_counter += 1

        extract_frame_counter += 1

    print(f'{saved_frame_counter} of {extract_frame_counter} frames saved')
    video_capture.release()


def extract_and_align_face_from_image(input_dir: Union[str, Path], desired_face_width: int=256) -> None:
    """
    Extract the face from an image, align it and save to a directory inside in the input directory
    Args:
        input_dir (str|Path): path to the directory contains the images extracted from a video
        desired_face_width (int): the width of the aligned imaged in pixels

    Returns:

    """

    input_dir = Path(input_dir)
    output_dir = input_dir / 'aligned'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()


    image = cv2.imread(str(input_dir / '00000.jpg'))
    image_height = image.shape[0]
    image_width = image.shape[1]

    detector = FaceExtractor((image_width, image_height))

    for image_path in tqdm(list(input_dir.glob('*.jpg'))):
        image = cv2.imread(str(image_path))

        ret, faces = detector.detect(image)
        if faces is None:
            continue

        face_aligned = detector.align(image, faces[0, :], desired_face_width)
        cv2.imwrite(str(output_dir / f'{image_path.name}'), face_aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])


class FaceExtractor:
    """A class to extract face from image using YuNet, and align it"""
    def __init__(self, image_size):
        """
        Create a YuNet face detector to get face from image of size 'image_size'. The YuNet model
        will be downloaded from opencv zoo, if it's not already exist.
        Args:
            image_size (tuple): a tuple of (width: int, height: int) of the image to be analyzed
        """
        detection_model_path = Path('models/face_detection_yunet_2023mar.onnx')
        if not detection_model_path.exists():
            detection_model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            print('Downloading face detection model...')
            filename, headers = urlretrieve(url, filename=str(detection_model_path))
            print('Download finish!')

        self.detector = cv2.FaceDetectorYN.create(str(detection_model_path), "", image_size)

    def detect(self, image):
        """
        Run the face detector
        Args:
            image (np.ndarray): an image with a face

        Returns:
            (int): return valvue indicate if the operation succeeded
            (np.ndarray): detection results of shape [num_faces, 15] (here num_faces=1)
                0-1: x, y of bbox top left corner
                2-3: width, height of bbox
                4-5: x, y of right eye
                6-7: x, y of left eye
                8-9: x, y of nose tip
                10-11: x, y of right corner of mouth
                12-13: x, y of left corner of mouth
                14: face score
        """
        ret, faces = self.detector.detect(image)
        return ret, faces

    @staticmethod
    def align(image, face, desired_face_width=256, left_eye_desired_coordinate=np.array((0.37, 0.37))):
        """
        Align the face so the eyes will be at the same level
        Args:
            image (np.ndarray): image with face
            face (np.ndarray):  face coordinates from the detection step
            desired_face_width (int): the final width of the aligned face image
            left_eye_desired_coordinate (np.ndarray): a length 2 array of values between
             0 and 1 where the left eye should be in the aligned image

        Returns:
            (np.ndarray): aligned face image
        """
        desired_face_height = desired_face_width
        right_eye_desired_coordinate = np.array((1 - left_eye_desired_coordinate[0], left_eye_desired_coordinate[1]))

        # get coordinate of the center of the eyes in the image
        right_eye = face[4:6]
        left_eye = face[6:8]

        # compute the angle of the right eye relative to the left eye
        dist_eyes_x = right_eye[0] - left_eye[0]
        dist_eyes_y = right_eye[1] - left_eye[1]
        dist_between_eyes = np.sqrt(dist_eyes_x ** 2 + dist_eyes_y ** 2)
        angles_between_eyes = np.rad2deg(np.arctan2(dist_eyes_y, dist_eyes_x) - np.pi)
        eyes_center = (left_eye + right_eye) // 2

        desired_dist_between_eyes = desired_face_width * (
                    right_eye_desired_coordinate[0] - left_eye_desired_coordinate[0])
        scale = desired_dist_between_eyes / dist_between_eyes

        M = cv2.getRotationMatrix2D(eyes_center, angles_between_eyes, scale)

        M[0, 2] += 0.5 * desired_face_width - eyes_center[0]
        M[1, 2] += left_eye_desired_coordinate[1] * desired_face_height - eyes_center[1]

        face_aligned = cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
        return face_aligned

    @staticmethod
    def extract(image, face, desired_face_width=256, left_eye_desired_coordinate=np.array((0.37, 0.37))):
        """
        Align the face so the eyes will be at the same level
        Args:
            image (np.ndarray): image with face
            face (np.ndarray):  face coordinates from the detection step
            desired_face_width (int): the final width of the aligned face image
            left_eye_desired_coordinate (np.ndarray): a length 2 array of values between
             0 and 1 where the left eye should be in the aligned image

        Returns:
            (np.ndarray): aligned face image
        """
        desired_face_height = desired_face_width

        # get coordinate of the center of the eyes in the image
        right_eye = face[4:6]
        left_eye = face[6:8]

        eyes_center = (left_eye + right_eye) // 2

        x_left = (eyes_center[0] - desired_face_width//2).astype(int)
        width = desired_face_width
        y_top = (eyes_center[1] - left_eye_desired_coordinate[1] * desired_face_height).astype(int)
        height = desired_face_height
        face = image[y_top:y_top+height, x_left:x_left+width]
        return face, [x_left, y_top, width, height]
