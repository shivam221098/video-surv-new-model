import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("saved_model_v1")
caption = cv2.VideoCapture("test_video_nv.mp4")
frame_rate = caption.get(5)
video = []

if caption.isOpened():
    while caption.isOpened():
        ret, frame = caption.read()

        if ret:
            for i in range(5):
                ret, frame = caption.read()
                if ret:
                    cv2.imshow("Frame", frame)
                    frame = cv2.resize(frame, (64, 64))
                    video.append(frame)

            frames = []
            for linear_sep in np.linspace(0, len(video) - 1, num=5):
                frames.append(video[int(linear_sep)])
            # result = model.predict(x)
            # print(result[0][0] * 100)
            # if (result[0][0] * 100) > 50:
            #     print("Violence")
            x = np.array(frames)
            x = np.expand_dims(x, axis=0)
            print(model.predict(x))
            video.clear()

            if cv2.waitKey(25) and 0xFF == ord('q'):
                break

        else:
            break

caption.release()
cv2.destroyAllWindows()
