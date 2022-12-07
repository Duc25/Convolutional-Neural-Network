from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

# imge_test=cv2.imread('1.jpg')
# imge_test=cv2.resize(imge_test,(32,32))
# imge_gray=cv2.cvtColor(imge_test, cv2.COLOR_BGR2GRAY)
(training_image, training_labels), (test_image, test_labels) = cifar10.load_data()
classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
training_image, test_image=training_image/255, test_image/255
training_labels_ohc, test_labels_ohc=to_categorical(training_labels), to_categorical(test_labels)
# for i in range (50000):
#     training_image[i]=cv2.resize(training_image[i],(224,244))
#     i=i+1

# model_first=models.Sequential([
#     layers.Conv2D(32,(3, 3),input_shape=(32,32,3), activation='relu'),
#     layers.Conv2D(32,(3, 3),input_shape=(32,32,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),

#     layers.Conv2D(64,(3, 3),input_shape=(32,32,3), activation='relu'),
#     layers.Conv2D(64,(3, 3),input_shape=(32,32,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),

#     layers.Conv2D(128,(3, 3),input_shape=(32,32,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),

#     # layers.Conv2D(256,(3, 3),input_shape=(32,32,3), activation='relu'),
#     # layers.MaxPool2D((2,2)),
#     # layers.Dropout(0.2),

   
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(3000, activation='relu'),
#     layers.Dense(3000, activation='relu'),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(10, activation='softmax'),
# ])

# model_first.compile(optimizer='adam', 
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'],
# )

# model_first.fit(training_image, training_labels_ohc,epochs=20)

# model_first.save('model_test.h5')
models=models.load_model('model_test.h5')


# pred=models.predict(test_image[1001].reshape((-1,32,32,3)))

# print(classes[np.argmax(pred)])
# plt.imshow(test_image[1001])
# plt.show()





# for i in range (50):
#     plt.subplot(5,10,i+1)
#     plt.imshow(training_image[100+i])
#     plt.title(classes[np.argmax(models.predict(test_image[100+i].reshape((-1,32,32,3))))])
#     plt.axis('off')
   
# plt.show()
