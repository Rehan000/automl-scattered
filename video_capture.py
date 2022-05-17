import cv2
import numpy as np


class VideoReader:
    def __init__(self, path, step_size=0, reshape_size=(512, 512)):
        self.path = path
        self.step_size = step_size
        self.curr_frame_no = 0
        self.video_finished = False
        self.reshape_size = reshape_size

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        return self

    def read(self):
        success, frame = self.cap.read()
        if not success:
            self.video_finished = True
            return success, frame
        for _ in range(self.step_size - 1):
            s, f = self.cap.read()
            if not s:
                self.video_finished = True
                break
        return success, frame

    def read_all(self):
        frames_list = []
        while not self.video_finished:
            success, frame = self.read()
            if success:
                # frame = resize(frame , self.reshape_size ) * 255).astype(np.uint8)
                frames_list.append(frame)
        return frames_list

    def __exit__(self, a, b, c):
        self.cap.release()
        cv2.destroyAllWindows()


def read_video(video_path, skip_step):
    """
    Parameters
    ----------
    video_path: string
            Path of video to be read
    skip_step: int
            No of frames to skip while reading the video
    Returns
    -------
            Numpy Array of All frames from the video in uint8 format
    """
    with VideoReader(path=video_path, step_size=skip_step) as reader:
        frames = reader.read_all()

    return np.array(frames, dtype=np.uint8)


def read_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
