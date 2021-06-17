# coding = utf-8
 
import cv2, json, os
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename  
from PIL import Image, ImageDraw, ImageFont
from nudenet import NudeClassifier
from videoprops import get_video_properties


app = Flask(__name__)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/static/<path:path>', methods=['GET'])
def static_dir(path):
    return send_from_directory('static', path)

@app.route('/upload/files', methods=['POST'])
def image_inference():
    uploadedFiles = request.files.getlist('file[]')
    original_files = []
    result = []

    for f in uploadedFiles:
        f.save('./files/'+secure_filename(f.filename))
        original_files.append('./files/'+secure_filename(f.filename))

    preds = classifier.classify(original_files, batch_size=32)

    for filename in preds:
        pred = preds.get(filename)

        if(pred['safe'] > pred['unsafe']):
            result.append({'unsafe': False})
        else:
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
            result.append({'unsafe': True, 'url': request.host_url+'static/'+secure_filename(f.filename)})

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