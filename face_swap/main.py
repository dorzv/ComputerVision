from pathlib import Path
import face_extraction_tools as fet
import quick96 as q96
from merge_frame_to_fake_video import merge_frames_to_fake_video

##### user parameters #####
# True for executing the step
extract_and_align_src = True
extract_and_align_dst = True
train = True
eval = False

model_name = 'Quick96'  # use this name to save and load the model
new_model = False  # True for creating a new model even if a model with the same name already exists

##### end of user parameters #####

# the path for the videos to process
data_root = Path('./data')
src_video_path = data_root / 'data_src.mp4'
dst_video_path = data_root / 'data_dst.mp4'

# path to folders where the intermediate product will be saved
src_processing_folder = data_root / 'src'
dst_processing_folder = data_root / 'dst'

# step 1: extract the frames from the videos
if extract_and_align_src:
    fet.extract_frames_from_video(video_path=src_video_path, output_folder=src_processing_folder, frames_to_skip=0)
if extract_and_align_dst:
    fet.extract_frames_from_video(video_path=dst_video_path, output_folder=dst_processing_folder, frames_to_skip=0)

# step 2: extract and align face from frames
if extract_and_align_src:
    fet.extract_and_align_face_from_image(input_dir=src_processing_folder, desired_face_width=256)
if extract_and_align_dst:
    fet.extract_and_align_face_from_image(input_dir=dst_processing_folder, desired_face_width=256)

# step 3: train the model
if train:
    q96.train(str(data_root), model_name, new_model, saved_models_dir='saved_model')

# step 4: create the fake video
if eval:
    merge_frames_to_fake_video(dst_processing_folder, model_name, saved_models_dir='saved_model')