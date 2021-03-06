import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("saved_model_v3")
caption = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
font_size = 1
thickness = 2

while caption.isOpened():
    ret, frame = caption.read()
    if ret:
        resized_frame = cv2.resize(frame, (64, 64))
        x = np.expand_dims(resized_frame, axis=0)
        preds = model.predict(x)
        perc = round(float(preds[0]) * 100, 2)
        if preds > 0.6:
            text = f"Violence Detected - {perc}%"
            color = (0, 0, 255)
        else:
            text = f"No Violence Detected - {perc}%"
            color = (0, 255, 0)

        frame = cv2.putText(frame, text, position, font, font_size, color, thickness, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
    else:
        break

    if cv2.waitKey(1) == ord('q'):
        break

caption.release()
cv2.destroyAllWindows()
