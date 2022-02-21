import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("saved_model_v2")
caption = cv2.VideoCapture("test_video.mp4")

while caption.isOpened():
    ret, frame = caption.read()
    if ret:
        cv2.imshow("Frame", frame)
        frame = cv2.resize(frame, (64, 64))
        x = np.expand_dims(frame, axis=0)
        preds = model.predict(x)
        if preds > 0.5:
            print("Violence")
        else:
            print("Non Violence")
    else:
        break

caption.release()
cv2.destroyAllWindows()