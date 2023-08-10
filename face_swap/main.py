from pathlib import Path
import face_extraction_tools as fet

# the path for the videos to process
data_root = Path('./data')
src_video_path = data_root / 'data_src.mp4'
dst_video_path = data_root / 'data_dst.mp4'

# path to folders where the intermediate product will be saved
src_processing_folder = Path('./src_processing')
dst_processing_folder = Path('./dst_processing')

# step 1: extract the frames from the videos
fet.extract_frames_from_video(video_path=src_video_path, output_folder=src_processing_folder, frames_to_skip=0)
fet.extract_frames_from_video(video_path=dst_video_path, output_folder=dst_processing_folder, frames_to_skip=0)

# step 2: extract and align face from frames
fet.extract_and_align_face_from_image(input_dir=src_processing_folder, desired_face_width=256)
fet.extract_and_align_face_from_image(input_dir=dst_processing_folder, desired_face_width=256)