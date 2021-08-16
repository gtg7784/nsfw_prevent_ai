# coding = utf-8

import cv2
import json
import os
import base64
import string
import random
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from nudenet import NudeClassifier
from io import BytesIO

app = Flask(__name__)
CORS(app)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/classify', methods=['POST'])
def b64_image_inference():
    image = request.files['image']

    ascii_letters = list(string.ascii_letters)
    random.shuffle(ascii_letters)
    filename = ''.join(ascii_letters[:8]) + '.jpg'
    filepath = f"./static/{filename}"
    image.save(filepath)

    preds = classifier.classify(filepath, batch_size=32)

    pred = preds.get(filepath)

    if(pred['safe'] > pred['unsafe']):
        result_image = filename
    else:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = cv2.blur(img,(250, 250))
        dstImg = Image.fromarray(dst)
        draw = ImageDraw.Draw(dstImg)
        font = ImageFont.truetype('./font/SpoqaHanSansNeo-Regular.ttf', 60)
        msg = '검열된 이미지'
        W, H = dstImg.width, dstImg.height
        w, h = draw.textsize(msg, font=font)
        draw.text(((W-w)/2, (H-h)/2), msg, (255, 0, 0), font=font)
        dst_filename = f'censored_{filename}'

        dstImg.save(f'./static/{dst_filename}')

        result_image = dst_filename

    return send_from_directory('static', result_image)

if __name__ == '__main__':
    app.run()
