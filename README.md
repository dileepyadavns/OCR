# OCR


step1: Downloaded Images for training and validation and labelled them for training on yolo v5

step2: will extract the number plate from the images 

step3: later will apply OCR on the number plate to extract the text on number plate

step4: Creating table on postgreSQL and will give some number plate details

step5: later will give random image to ocr and will predict the text and check the content whether it is present in postgreSQL if it is present then it will print on the webpage as acess granted if not present will print as acess not granted


# best.pt is the weight file i got after training the yolov5 on custom data here i used this file to detect the images 
