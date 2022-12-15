import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# from cvzone.SerialModule import SerialObject 
from tensorflow.keras import layers
from tensorflow.keras import models

# Arduino = SerialObject('COM4',9600,1) 
# wpTypeArr = np.array([0],dtype=int) 
# wpTypeArr[0] = 2

# input image processing
TRAIN_DATA = 'Datasets/Train_data'
TEST_DATA = 'Datasets/Test_data'

Train_image = []
Train_label = []

Test_image = []
Test_label = []

dict = {'trainKTron':[1,0,0], 'trainKTru':[0,1,0], 'trainKVuong':[0,0,1],
        'testKTron':[1,0,0], 'testKTru':[0,1,0], 'testKVuong':[0,0,1]}

def getData(dataGet, lstData):
    for whatever in os.listdir(dataGet):
        whatever_path = os.path.join(dataGet, whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            label = filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            img_resize = cv2.resize(img,(224,224))
            lst_filename_path.append((img_resize, dict[label]))
        lstData.extend(lst_filename_path)
    return lstData

Train_image = getData(TRAIN_DATA, Train_image)
Test_image = getData(TEST_DATA, Test_image)
# Train models
# model_first=models.Sequential([
#     layers.Conv2D(64,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(64,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(128,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(128,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Flatten(input_shape=(224,224,3)),
#     layers.Dense(4096, activation='relu'),
#     layers.Dense(4096, activation='relu'),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(3, activation='softmax'),
# ])
#
# # model_first.summary()
# model_first.compile(optimizer='adam',
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'],
# )
#
# model_first.fit(np.array([x[0] for i ,x in enumerate(Train_image)]),
#                 np.array([y[1] for i ,y in enumerate(Train_image)]),
#                 epochs=20)
#
# model_first.save('main_model.h5')
models=models.load_model('main_model.h5')
# pred=models.predict(Test_image[20][0].reshape((-1,224,224,3)))

# print(dict[np.argmax(pred)])
# plt.imshow(Test_image[20][0])
# plt.show()
# read cam
lstResult = ['Khoi Tron', 'Khoi Tru', 'Khoi Vuong']
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

for i in range(10):
    _,frame = cap.read()
frame = cv2.resize(frame, (640,480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(25,25), 0)
last_frame = gray
count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    abs_img = cv2.absdiff(last_frame, gray)
    last_frame = gray
    _, img_mask = cv2.threshold(abs_img, 30,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 900:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        print(type(x))
        roi = cv2.resize(frame[y:y+h, x:x+w],(224,224))
        # cv2.imwrite('Datasets/anh1_{}.jpg'.format(count),frame[y:y+h, x:x+w])
        result =np.argmax(models.predict(roi.reshape((-1, 224, 224, 3))))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, lstResult[result], (x+15, y+15), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)
        count += 1
    # if result == 0:
    #     wpTypeArr[0] = 1
    # elif result == 1:
    #     wpTypeArr[0] = 2
    # elif result == 2:
    #     wpTypeArr[0] = 3
    # Arduino.sendData(wpTypeArr)
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break