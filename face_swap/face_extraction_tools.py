import cv2
from pathlib import Path

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
            cv2.imwrite(str(output_folder / f'{saved_frame_counter:05d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved_frame_counter += 1

        extract_frame_counter += 1

    print(f'{saved_frame_counter} of {extract_frame_counter} frames saved')