import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../models/saved_model_v3")
caption = cv2.VideoCapture(1)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
font_size = 1
thickness = 2
foreground_background = cv2.createBackgroundSubtractorMOG2()

while caption.isOpened():
    ret, frame = caption.read()
    if ret:
        resized_frame = cv2.resize(frame, (64, 64))
        x = np.expand_dims(resized_frame, axis=0)
        preds = model.predict(x)
        if preds > 0.5:
            text = "Violence Detected"
            color = (0, 0, 255)
        else:
            text = "No Violence Detected"
            color = (0, 255, 0)

        # frame = cv2.putText(frame, text, position, font, font_size, color, thickness, cv2.LINE_AA)
        mask = foreground_background.apply(frame)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        _, th = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(th, cv2.CV_32SC1, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1000 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Frame", mask)

    else:
        break

    if cv2.waitKey(1) == ord('q'):
        break

caption.release()
cv2.destroyAllWindows()