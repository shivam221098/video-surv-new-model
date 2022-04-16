import cv2
import numpy as np
import tensorflow as tf
from frame_extraction import Frames

model = tf.keras.models.load_model("saved_model_v1")
caption = cv2.VideoCapture("test_video.mp4")
print(model.predict(Frames.select_fr(Frames.read_fr(["test_video_nv.mp4"]))))
frame_rate = caption.get(5)
video_frames = []

# if caption.isOpened():
#     while caption.isOpened():
#         i = 0
#         while i < 5:
#             ret, frame = caption.read()
#             frame_id = caption.get(1)
#             if ret:
#                 # cv2.imshow("Video", frame)
#                 resized_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
#                 video_frames.append(resized_frame)
#                 i += 1
#
#             else:
#                 break
#
#         if len(video_frames) > 0:
#             numpy_video = []
#             for linear_sep in np.linspace(0, len(video_frames) - 1, num=5):
#                 numpy_video.append(video_frames[int(linear_sep)])
#
#             video_frames.clear()
#             np_video = np.array(numpy_video)
#             np_video = np.expand_dims(np_video, axis=0)
#             print(model.predict(np_video))
#
# caption.release()
# cv2.destroyAllWindows()
# x = np.array(video)
# x = np.expand_dims(x, axis=0)
# print(model.predict(x))
appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor