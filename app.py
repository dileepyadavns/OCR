import torch
import io
from PIL import Image
import cv2
import argparse
import imutils
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

model = torch.hub.load( '/home/neosoft/Documents/OCR/yolov5','custom',path='/home/neosoft/Documents/OCR/best.pt',source='local') # for PIL/cv2/np inputs and

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)  
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
    
        crops = results.crop(save=True)  
        
        crops=crops[0]['im']
        #ad
        img = cv2.cvtColor(crops, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(img, 0, 255,
	    # cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
#     	cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         chars = []
# # loop over the contours
#         for c in cnts:
#              (x, y, w, h) = cv2.boundingRect(c)
#              if w >= 35 and h >= 100:
#                  chars.append(c)

#         chars = np.vstack([chars[i] for i in range(0, len(chars))])
#         hull = cv2.convexHull(chars)
#         # allocate memory for the convex hull mask, draw the convex hull on
#         # the image, and then enlarge it via a dilation
#         mask = np.zeros(crops.shape[:2], dtype="uint8")
#         cv2.drawContours(mask, [hull], -1, 255, -1)
#         mask = cv2.dilate(mask, None, iterations=2)
#         cv2.imshow("Mask", mask)
# # take the bitwise of the opening image and the mask to reveal *just*
# # the characters in the image
#         img = cv2.bitwise_and(img, img, mask=mask )       
        #ad
        img=Image.fromarray(img)
        img.show()
        text = pytesseract.image_to_string(img)
        return render_template('results.html',path=text)   

    
    return render_template('index.html')    
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5500)
