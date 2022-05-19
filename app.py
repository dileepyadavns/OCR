#Importing libraries
import torch
import io
from PIL import Image
import cv2
import numpy as np
import imutils
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
model = torch.hub.load( '/home/neosoft/Documents/OCR/yolov5','custom',path='/home/neosoft/Documents/OCR/best2.pt',source='local') # for PIL/cv2/np inputs and

#to detect the number plate
def number_plate_detection(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)  
    return results

#home page of flask 
@app.route('/', methods=['GET', 'POST'])
def number_plate_extraction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file') #to take image from user
        if not file:
            return

        img_bytes = file.read()
        number_plate_with_bounding_boxes = number_plate_detection(img_bytes)
        #results.show()
    
        cropped_image_of_number_plate = number_plate_with_bounding_boxes.crop(save=True) #cropping the image on bounding box
        img=cropped_image_of_number_plate[0]['im'] #to get the image array from the array created from above step
        img = imutils.resize(img, width=300 )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to convert image to gray
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        img = cv2.bilateralFilter(img, 9, 75, 75) 
        img = cv2.medianBlur(img, 3)
        

        cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
          
         
        # Apply dilation and erosion to remove some noise    
        kernel = np.ones((1, 1), np.uint8)    
        img = cv2.dilate(img, kernel, iterations=1)    
        img = cv2.erode(img, kernel, iterations=1)

        img=Image.fromarray(img) #to get image from array
        img.show()#to display image
        
        text = pytesseract.image_to_string(img)#to get the charcters from number plate
        
        #for removing empty spaces
        text1='' 
        for i in text:
            if i==" ":
                continue
            else:
                text1+=i
        text2=''        
        for i in text1:
            if i.lower()!=i or i.isdigit():
                text2+=i        
        # if i[0].isdigit:
        #     text2=text2[1:]
        # else:
        #     text2=text2    

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        s = "SELECT * FROM cars where vehicleNo =%s"

        cur.execute(s, (text2,))
        
        res = cur.fetchall()
        if len(res)>=1:
            return render_template('results.html',path=text2,msg="Car has granted permission",cls='green',symbol="static/right.png")#to redirect to results html page   

        else:
            return render_template('results.html',path=text2,msg="Car has not granted permission",cls='red',symbol="static/wrong.jpeg")#to redirect to results html page   

    return render_template('index.html')    
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5600)
