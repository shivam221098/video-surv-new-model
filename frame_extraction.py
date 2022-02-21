import cv2
import numpy as np


class Frames:
    fps = 5

    @classmethod
    def read_fr(cls, arr):
        """
        method for reading frames from videos
        :return: array of video frames
        """
        videos = []
        for index, video in enumerate(arr):
            print("Completed", round(index / len(arr) * 100, 2), end="\r")
            resized_frames = []

            caption = cv2.VideoCapture(video)
            frame_rate = caption.get(Frames.fps)

            while caption.isOpened():
                frame_id = caption.get(1)
                ret, frame = caption.read()

                if not ret:
                    break

                if frame_id % np.floor(frame_rate) == 0:
                    resized_frames.append(cv2.resize(frame, (64, 64)))

            videos.append(resized_frames)
            caption.release()

        return videos

    @classmethod
    def select_fr(cls, arr):
        """
        method for selecting frames
        """
        videos = []
        for i in range(len(arr)):
            frames = []
            for linear_sep in np.linspace(0, len(arr[i]) - 1, num=Frames.fps):
                frames.append(arr[i][int(linear_sep)])

            if len(frames) <= 5:
                videos.append(frames)

        np_videos = np.array(videos)
        return np_videos
