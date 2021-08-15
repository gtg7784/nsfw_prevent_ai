# coding = utf-8

import cv2
import json
import os
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from videoprops import get_video_properties
from nudenet import NudeClassifier
import base64
import string
import random
from io import BytesIO

app = Flask(__name__)
CORS(app)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/classify', methods=['POST'])
def b64_image_inference():
    b64_data = request.json['image']

    b64_image = b64_data.split(',')[1]
    image_info = b64_data.split(',')[0]
    image_type = image_info.split('image/')[1][:-7]

    ascii_letters = list(string.ascii_letters)
    random.shuffle(ascii_letters)
    filedata = base64.b64decode(b64_image)
    filename = ''.join(ascii_letters[:8]) + '.' + image_type
    filepath = f"./files/{filename}"
    with open(filepath, 'wb') as f:
        f.write(filedata)

    preds = classifier.classify(filepath, batch_size=32)

    pred = preds.get(filepath)

    if(pred['safe'] > pred['unsafe']):
        result_image = b64_image
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

        buffered = BytesIO()
        dstImg.save(buffered, format="JPEG")
        result_image = base64.b64encode(buffered.getvalue()).decode()

        print(result_image)

    response = {
        'image': image_info +',' + result_image,
        'pred': pred
    }

    os.remove(filepath)

    response = app.response_class(
        response=json.dumps(response),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/upload/files', methods=['POST'])
def image_inference():
    uploadedFiles = request.files.getlist('file[]')
    original_files = []
    result = []

    for f in uploadedFiles:
        f.save('./files/'+secure_filename(f.filename))
        original_files.append('./files/'+secure_filename(f.filename))

    preds = classifier.classify(original_files, batch_size=32)

    for index, filename in enumerate(preds):
        pred = preds.get(filename)

        if(pred['safe'] > pred['unsafe']):
            result.append({'unsafe': False})
        else:
            f = uploadedFiles[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dst = cv2.blur(img,(250, 250))
            dstImg = Image.fromarray(dst)
            draw = ImageDraw.Draw(dstImg)
            font = ImageFont.truetype('./font/SpoqaHanSansNeo-Regular.ttf', 60)
            msg = 'Censored'
            W, H = dstImg.width, dstImg.height
            w, h = draw.textsize(msg, font=font)
            draw.text(((W-w)/2, (H-h)/2), msg, (255, 0, 0), font=font)
            dstImg.save('./static/'+secure_filename(f.filename))
            
            result.append({'unsafe': True, 'url': f'{request.host_url}/static/{secure_filename(f.filename)}'})

        os.remove(filename)

    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/upload/video', methods=['POST'])
def video_inference():
    f = request.files['file']
    path = './files/'+secure_filename(f.filename)
    filename = f.filename.split('.')[0]

    f.save(path)

    props = get_video_properties(path)
    W, H = int(props['width']), int(props['height'])
    
    preds = classifier.classify_video(path)
    preds = preds.get('preds')
    unsafe, safe = 0, 0

    for pred in preds:
        unsafe += preds.get(pred)['unsafe']
        safe += preds.get(pred)['safe']

    if safe > unsafe:
        result = {
            'unsafe': False
        }
    else:
        img = Image.new('RGB', (W, H), color='black')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./font/SpoqaHanSansNeo-Regular.ttf', 60)
        msg = 'Censored'
        w, h = draw.textsize(msg, font=font)
        draw.text(((W-w)/2, (H-h)/2), msg, (255, 0, 0), font=font)
        img.save(f'./static/{filename}.png')
        result = {
            'unsafe': True,
            'url': request.host_url+'static/'+filename+'.png'
        }

    os.remove(path)

    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run()
