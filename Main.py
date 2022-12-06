import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from openni import openni2
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

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
# print(Train_image[30])
# Train models
model_first=models.Sequential([
    layers.Conv2D(64,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(64,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(128,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(128,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(256,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.Conv2D(512,(3, 3),input_shape=(224,224,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(input_shape=(224,224,3)),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(3, activation='softmax'),
])

# model_first.summary()
model_first.compile(optimizer='adam', 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
)

model_first.fit(np.array([x[0] for i ,x in enumerate(Train_image)]),
                np.array([y[1] for i ,y in enumerate(Train_image)]), 
                epochs=20)

model_first.save('main_model.h5')
models=models.load_model('main_model.h5')
pred=models.predict(Test_image[20][0].reshape((-1,224,224,3)))

print(dict[np.argmax(pred)])
plt.imshow(Test_image[20][0])
plt.show()



# shapes_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')
# cam = cv2.VideoCapture('video_test.mp4')

# while True:
#     OK, frame = cam.read()
#     shapes = shapes_detector.detectMultiScale(frame, 1.3, 5)
    
#     for (x, y, w, h) in shapes:
#         roi = cv2.resize(frame[y: y+h, x:x+w], (224,224))
#         result = models.predict(roi.reshape((-1, 224, 224, 3)))
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (224, 224, 50), 1)
#     cv2.imshow('FRAME', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()
def show_depth_value(event, x, y, flags, param):
    global depth
    print(depth[y, x])
    
if __name__ == '__main__':
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    color_stream = dev.create_color_stream()
    color_stream.start()
    depth_scale_factor = 255.0 / depth_stream.get_max_pixel_value()
    print (depth_stream.get_max_pixel_value())
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', show_depth_value)

    while True:
        # Get depth
        # depth_frame = depth_stream.read_frame()       
        # h, w = depth_frame.height, depth_frame.width
        # depth = np.ctypeslib.as_array(
        #     depth_frame.get_buffer_as_uint16()).reshape(h, w)     
        # depth_uint8 = cv2.convertScaleAbs(depth, alpha=depth_scale_factor)
        # depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_HSV)
        
        # Get color
        color_frame = color_stream.read_frame()
        color = np.ctypeslib.as_array(color_frame.get_buffer_as_uint8()).reshape(h, w, 3)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Display
        cv2.imshow('depth', depth_uint8)
        cv2.imshow('depth colored', depth_colored)
        cv2.imshow('color', color)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    depth_stream.stop()
    openni2.unload()
