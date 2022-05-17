from typing import Text
from requests import patch
import torch
import io
from PIL import Image
import cv2
import numpy as np

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from itsdangerous import Signer, BadSignature, want_bytes
import psycopg2 #pip install psycopg2 
import psycopg2.extras
import pytesseract
conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
        host="localhost",
        database="students",
        user='postgres',
        password='p@ssw0rd')

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load( '/home/neosoft/Documents/OCR/yolov5','custom',path='/home/neosoft/Documents/OCR/best.pt',source='local') # for PIL/cv2/np inputs and

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
     # batched list of images

# Inference
    results = model(img, size=640)  # includes NMS
    return results


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        print(type(results))
        results.save('results0.jpg')
        # path=''
        
        # for file in os.listdir("runs/detect"):
        #     path='runs/detect/{}/image0.jpg'.format(file)

        crops = results.crop(save=True)  
        #cv2.imshow('croped',crops[0]['im'])
        crops=crops[0]['im']
        #array1=np.array(crops)
        #var1=array1.astype(np.uint8)
        img=Image.fromarray(crops)

        img.show()
        text = pytesseract.image_to_string(img)
        return render_template('results.html',path=text)   

         # save as results1.jpg, results2.jpg... etc.
        #os.rename("results0.jpg", "static/results0.jpg")
        # results.show()
        
        # # 
        # # cv2.imwrite("crop.jpg",crops)
        # # crops.show()
        # 
        # # cv2.imshow('croped',array1)
        # # print(array1)
        #ull_filename = os.path.join(app.config['RESULT_FOLDER'], 'results0.jpg')
        #return redirect(full_filename)
    return render_template('index.html')    
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5500)
