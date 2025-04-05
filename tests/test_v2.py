import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.models import Model

model = tf.keras.models.load_model("../models/saved_model_v2")
vgg_model = VGG16(include_top=True, weights="imagenet")
transfer_layer = vgg_model.get_layer("fc2")
image_model_transfer = Model(inputs=vgg_model.input, outputs=transfer_layer.output)

caption = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
font_size = 1
thickness = 2

batch = []
batch_size = 1
np.set_printoptions(threshold=np.inf)
classes = ['fight', "no fight"]

text = f"No Violence Detected"
color = (0, 255, 0)

while caption.isOpened():
    ret, frame = caption.read()

    if ret:
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        x = np.expand_dims(resized_frame, axis=0)
        if batch_size % 21 == 0:
            y = np.array(batch)
            z = np.resize(y, (20, 4096))
            k = np.expand_dims(z, axis=0)
            print(classes[np.argmax(model.predict(k))])
            if np.argmax(model.predict(k)) == 0:
                text = f"Violence Detected"
                color = (0, 0, 255)
            else:
                text = f"No Violence Detected"
                color = (0, 255, 0)
            batch.clear()
        else:
            batch.append(image_model_transfer.predict(x))

        frame = cv2.putText(frame, text, position, font, font_size, color, thickness, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
    else:
        break

    if cv2.waitKey(1) == ord('q'):
        break

    batch_size += 1

caption.release()
cv2.destroyAllWindows()
