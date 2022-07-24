import numpy as np
import cv2
from LK import lucas_kanade_pyramid, wrap_image


def lucas_kanade_video_stabilizer(video_path, window_size, maximum_iteration, number_of_levels):
    video_cap = cv2.VideoCapture(video_path)
    number_of_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, previous_frame = video_cap.read()
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    stabilized_video = cv2.VideoWriter('data/stabilized_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (int(video_cap.get(3)), int(video_cap.get(4))))
    stabilized_video.write(previous_frame)
    rows, cols = previous_frame.shape
    u = np.zeros((rows, cols))
    v = np.zeros((rows, cols))

    for ii in range(2, number_of_frames + 1):
        print(f'Working on frame {ii}/{number_of_frames}')
        ret, current_frame = video_cap.read()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame = current_frame / 255.
        du, dv = lucas_kanade_pyramid(previous_frame, current_frame, window_size, maximum_iteration, number_of_levels)
        u = u + du
        v = v + dv
        wrap = wrap_image(current_frame, u, v)
        wrap = cv2.normalize(wrap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        wrap = cv2.cvtColor(wrap, cv2.COLOR_GRAY2BGR)
        stabilized_video.write(wrap)
        previous_frame = current_frame
    video_cap.release()
    stabilized_video.release()


if __name__ == '__main__':
    lucas_kanade_video_stabilizer('data/inputVid.avi', 100, 200, 4)
    # im = 255 * np.ones((200, 200), np.uint8)
    # stabilized_video = cv2.VideoWriter('data/stabilized_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24,
    #                                    (200, 200))
    # for ii in range(100):
    #     stabilized_video.write(im)
    # stabilized_video.release()