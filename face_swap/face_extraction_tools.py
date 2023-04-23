import cv2
from pathlib import Path
from shutil import rmtree
from urllib.request import urlretrieve
import numpy as np
from tqdm import tqdm

def extract_frames_from_video(video_path: str, output_folder: str, frames_to_skip: int=0) -> None:
    """
    Extract frame from video as a JPG images.
    Args:
        video_path (str): the path to the input video from it the frame will be extracted
        output_folder (str): the folder where the frames will be saved
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


def extract_and_align_face_from_image(input_dir: str, desired_face_width: int=256) -> None:
    """
    Extract the face from an image, align it and save to a directory inside in the input directory
    Args:
        input_dir (str): path to the directory contains the images extracted from a video
        desired_face_width (int): the width of the aligned imaged in pixels

    Returns:

    """

    input_dir = Path(input_dir)
    output_dir = input_dir / 'aligned'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    desired_face_height = desired_face_width

    image = cv2.imread(str(input_dir / '00000.jpg'))
    image_height = image.shape[0]
    image_width = image.shape[1]

    detection_model_path = Path('models/face_detection_yunet_2022mar.onnx')
    if not detection_model_path.exists():
        detection_model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
        print('Downloading face detection model...')
        filename, headers = urlretrieve(url, filename=str(detection_model_path))
        print('Download finish!')

    detector = cv2.FaceDetectorYN.create(str(detection_model_path), "", (image_width, image_height))

    left_eye_desired_coordinate = np.array((0.37, 0.37))
    right_eye_desired_coordinate = np.array((1 - left_eye_desired_coordinate[0], left_eye_desired_coordinate[1]))

    for image_path in tqdm(list(input_dir.glob('*.jpg'))):
        image = cv2.imread(str(image_path))

        ret, faces = detector.detect(image)
        if faces is None:
            continue

        # align the face
        # get coordinate of the center of the eyes in the image
        right_eye = faces[0, 4:6]
        left_eye = faces[0, 6:8]

        # compute the angle of the right eye relative to the left eye
        dist_eyes_x = right_eye[0] - left_eye[0]
        dist_eyes_y = right_eye[1] - left_eye[1]
        dist_between_eyes = np.sqrt(dist_eyes_x ** 2 + dist_eyes_y ** 2)
        angles_between_eyes = np.rad2deg(np.arctan2(dist_eyes_y, dist_eyes_x) - np.pi)
        eyes_center = (left_eye + right_eye) // 2

        desired_dist_between_eyes = desired_face_width * (right_eye_desired_coordinate[0] - left_eye_desired_coordinate[0])
        scale = desired_dist_between_eyes / dist_between_eyes

        M = cv2.getRotationMatrix2D(eyes_center, angles_between_eyes, scale)

        M[0, 2] += 0.5 * desired_face_width - eyes_center[0]
        M[1, 2] += left_eye_desired_coordinate[1] * desired_face_height - eyes_center[1]

        face_aligned = cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)

        cv2.imwrite(str(output_dir / f'{image_path.name}'), face_aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])

