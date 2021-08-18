# coding = utf-8

import os
import cv2
import string
import random
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from nudenet import NudeClassifier
import hashlib
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)
CORS(app)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/classify', methods=['POST'])
def b64_image_inference():
    image = request.files['image']
    content_type = request.form['content_type']

    if(content_type.split('/')[0] != 'image'):
        return app.response_class(status=400)

    filebytes = image.stream.read()

    md5 = hashlib.md5(filebytes)

    file_extention = content_type.split('/')[-1]
    filename = md5.hexdigest() + '.' + file_extention
    filepath = f"./static/{filename}"

    if filename in os.listdir('static'):
        return send_from_directory('static', filename)

    with open(filepath, 'wb') as f:
        f.write(filebytes)

    preds = classifier.classify(filepath, batch_size=32)

    pred = preds.get(filepath)

    app.logger.info('%s failed to log in', pred)

    if(pred['safe'] > pred['unsafe']):
        return send_from_directory('static', filename)

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

    os.remove(filepath)

    dstImg.save(f'./static/{filename}')

    return send_from_directory('static', filename)

def remove_all_files():
    files = os.listdir('static')
    for file in files:
        os.remove(f'./static/{file}')

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=remove_all_files, trigger="interval", seconds=86400)
    scheduler.start()

    atexit.register(lambda: scheduler.shutdown())

    app.run(host='0.0.0.0', port=80)
