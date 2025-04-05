from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from typing import List
import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.models import Model
import concurrent.futures
import datetime
import zipfile


model = tf.keras.models.load_model("models/saved_model_v3")
vgg_model = VGG16(include_top=True, weights="imagenet")
transfer_layer = vgg_model.get_layer("fc2")
image_model_transfer = Model(inputs=vgg_model.input, outputs=transfer_layer.output)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
font_size = 1
thickness = 2

np.set_printoptions(threshold=np.inf)
classes = ['fight', "no fight"]

try:
    os.mkdir("received")
except FileExistsError:
    pass

try:
    os.mkdir("processed_files")
except FileExistsError:
    pass


def prediction(batch):
    transfer_values = []
    for frame in batch:
        transfer_values.append(image_model_transfer.predict(frame))

    array = np.array(transfer_values)
    resized = np.resize(array, (20, 4096))
    expanded = np.expand_dims(resized, axis=0)
    preds = model.predict(expanded)
    print(classes[np.argmax(preds)])

    if np.argmax(preds) == 0:
        return batch
    return


def detect(filename):
    batch = []
    batch_size = 1
    violent_frames = []
    new_file_name = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    caption = cv2.VideoCapture(filename)
    width = int(caption.get(3))
    height = int(caption.get(4))

    size = (width, height)

    frames = []

    # while caption.isOpened():
    #     ret, frame = caption.read()
    #     if ret:
    #         frames.append(frame)
    #     else:
    #         break
    #
    # i = 0
    # futures = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     while i < len(frames):
    #         required_frames = list(map(lambda x: np.expand_dims(cv2.resize(x, (224, 224),
    #                                                                        interpolation=cv2.INTER_CUBIC), axis=0),
    #                                    frames[i: i + 20]))
    #         if len(required_frames) == 20:
    #             futures.append(executor.submit(prediction, required_frames))
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #
    #         if result is not None:
    #             violent_frames.append(result)


    while caption.isOpened():
        ret, frame = caption.read()
        if ret:
            resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            x = np.expand_dims(resized_frame, axis=0)
            if batch_size % 21 == 0:
                y = np.array(batch)
                z = np.resize(y, (20, 4096))
                k = np.expand_dims(z, axis=0)
                preds = model.predict(k)
                print(classes[np.argmax(preds)])
                if np.argmax(preds) == 0:
                    violent_frames.append(frames)
                batch.clear()
                frames.clear()
            else:
                batch.append(image_model_transfer.predict(x))
                frames.append(frame)
        else:
            break

        batch_size += 1

    file_names = []
    for index, frames in enumerate(violent_frames):
        writer = cv2.VideoWriter(f"processed_files/{new_file_name}_{index}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
        file_names.append(f"processed_files/{new_file_name}_{index}.avi")
        for frame in frames:
            writer.write(frame)
        writer.release()

    caption.release()
    cv2.destroyAllWindows()

    # zipping file
    with zipfile.ZipFile(f"processed_files/{new_file_name}.zip", "w") as zip_obj:
        for file in file_names:
            zip_obj.write(file)
            os.remove(file)

    os.remove(filename)

    return f"processed_files/{new_file_name}.zip"


app = FastAPI()
templates = Jinja2Templates(directory="templates")

received_path = "received"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "username": "Detect Violence in Videos"})


@app.post("/submitform")
async def upload_file(request: Request, uploaded_files: List[UploadFile] = File(...)):
    filenames = []
    for uploaded_file in uploaded_files:
        if uploaded_file.filename.endswith(".mp4"):
            filenames.append(uploaded_file.filename)
            with open(f"received/{uploaded_file.filename}", "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

    for files in os.walk("received"):
        for file in files[-1]:
            zip_file_name = detect(f"received/{file}")
            print(zip_file_name)

    return FileResponse(path=zip_file_name, filename=zip_file_name.split("/")[-1], media_type="application/octet-stream")
