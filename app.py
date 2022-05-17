#Importing libraries
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
import pytesseract #for number plate extraction

conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
        host="localhost",
        database="students",
        user='postgres',
        password='p@ssw0rd')

app = Flask(__name__)

#Loading the yolov5 model with the trained weights of  custom dataset
model = torch.hub.load( '/home/neosoft/Documents/OCR/yolov5','custom',path='/home/neosoft/Documents/OCR/best.pt',source='local') # for PIL/cv2/np inputs and

#to detect the number plate
def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)  
    return results

#home page of flask 
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file') #to take image from user
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        print(type(results))
        results.save('results0.jpg')
    
        crops = results.crop(save=True) #cropping the image on bounding box
        crops=crops[0]['im'] #to get the image array from the array created from above step
       
        img = cv2.cvtColor(crops, cv2.COLOR_BGR2GRAY) #to convert image to gray
        img=Image.fromarray(img) #to get image from array
        img.show()#to display image
        text = pytesseract.image_to_string(img)#to get the charcters from number plate
        return render_template('results.html',path=text)#to redirect to results html page   

    
    return render_template('index.html')    
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5500)
