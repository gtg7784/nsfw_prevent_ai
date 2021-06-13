# coding = utf-8
 
import cv2, json, os
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename  
from PIL import Image, ImageDraw, ImageFont
from nudenet import NudeClassifier


app = Flask(__name__)

classifier = NudeClassifier()

@app.route('/')
def home():
    return 'hello'

@app.route('/static/<path:path>', methods=['GET'])
def static_dir(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def file_upload():
    uploadedFiles = request.files.getlist('file[]')
    original_files = []
    result = []

    for f in uploadedFiles:
        f.save('./files/'+secure_filename(f.filename))
        original_files.append('./files/'+secure_filename(f.filename))

    preds = classifier.classify(original_files, batch_size=32)

    print(preds)

    for filename in preds:
        pred = preds.get(filename)

        if(pred['safe'] > pred['unsafe']):
            result.append({'unsafe': False})
        else:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dst = cv2.blur(img,(150, 150))
            dstImg = Image.fromarray(dst)
            draw = ImageDraw.Draw(dstImg)
            font = ImageFont.truetype('./font/SpoqaHanSansNeo-Regular.ttf', 60)
            msg = 'Censored'
            W, H = dstImg.width, dstImg.height
            w, h = draw.textsize(msg, font=font)
            draw.text(((W-w)/2, (H-h)/2), msg, (255, 0, 0), font=font)
            dstImg.save('./static/'+secure_filename(f.filename))
            result.append({'unsafe': True, 'url': request.host_url+'/static/'+secure_filename(f.filename)})

        os.remove(filename)

    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response
  

if __name__ == '__main__':
    app.run()