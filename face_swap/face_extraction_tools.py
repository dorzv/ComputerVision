import cv2
from pathlib import Path
from shutil import rmtree
from urllib.request import urlretrieve
from tqdm import tqdm

def extract_frames_from_video(video_path: str, output_folder: str, frames_to_skip: int=0):
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


def extract_and_align_face_from_image(input_dir: str):
    """
    Extract the face from an image, align it and save to a directory inside in the input directory
    Args:
        input_dir (str): path to the directory contains the images extracted from a video

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

    detection_model_path = Path('models/face_detection_yunet_2022mar.onnx')
    if not detection_model_path.exists():
        detection_model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
        print('Downloading face detection model...')
        filename, headers = urlretrieve(url, filename=str(detection_model_path))
        print('Download finish!')

    recognition_model_path = Path('models/face_recognition_sface_2021dec.onnx')
    if not recognition_model_path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        print('Downloading face recognition model...')
        filename, headers = urlretrieve(url, filename=str(recognition_model_path))
        print('Download finish!')

    detector = cv2.FaceDetectorYN.create(str(detection_model_path), "", (image_width, image_height))
    recognizer = cv2.FaceRecognizerSF.create(str(recognition_model_path), "")

    for image_path in tqdm(list(input_dir.glob('*.jpg'))):
        image = cv2.imread(str(image_path))

        ret, faces = detector.detect(image)
        if faces is None:
            continue
        face_aligned = recognizer.alignCrop(image, faces[0])
        cv2.imwrite(str(output_dir / f'{image_path.name}'), face_aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
