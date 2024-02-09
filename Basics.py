import cv2
import numpy as np
import face_recognition

#imgElon = face_recognition.load_image_file('Images/Elon_Musk.jpeg') #ELON MUSK TRAIN
#imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgBill = face_recognition.load_image_file('Images/Bill_Gates.jpg') #BILL GATES TEST
imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Bill_Test.jpg') #BILL GATES TRAIN
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,255),2)

results = face_recognition.compare_faces([encodeBill],encodeTest)
faceDis = face_recognition.face_distance([encodeBill],encodeTest)
print(results , faceDis)

#cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Bill Gates',imgBill)
cv2.imshow('Bill Test',imgTest)
cv2.waitKey(0)