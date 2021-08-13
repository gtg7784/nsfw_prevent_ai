# coding = utf-8

import cv2
import json
import os
import glob
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from videoprops import get_video_properties
from nudenet import NudeClassifier
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pydload
import onnxruntime

app = Flask(__name__)
CORS(app)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/classify', methods=['POST'])
def b64_image_inference():
    b64_image = request.json['image']

    imgdata = base64.b64decode(b64_image)

    img = Image.open(BytesIO(imgdata))
    img = img.convert("RGB")
    img = np.asarray(img)

    x = np.asarray(img, dtype='float32')

    if len(x.shape) == 3:
        x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        x = x.reshape((1, x.shape[0], x.shape[1]))

    x /= 255

    img_array = np.asarray(x)

    print(img_array.shape)
    
    # f gets closed when you e
    preds = classifier.classify_image(img_array)

    response = app.response_class(
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
    home = os.path.expanduser("~")
    model_folder = os.path.join(home, ".NudeNet/")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    model_path = os.path.join(model_folder, os.path.basename(url))

    if not os.path.exists(model_path):
        print("Downloading the checkpoint to", model_path)
        pydload.dload(url, save_to_path=model_path, max_time=None)
    sess = onnxruntime.InferenceSession(model_path)
    
    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = sess.get_inputs()[0].type
    print("input type", input_type)
    print("output name", sess.get_outputs()[0].name)
    print("output shape", sess.get_outputs()[0].shape)
    print("output type", sess.get_outputs()[0].type)
    app.run()
