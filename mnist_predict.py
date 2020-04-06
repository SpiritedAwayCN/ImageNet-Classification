from tensorflow import keras
import numpy as np
import cv2

img = cv2.imread('test.png', 0)

img = np.reshape(img, (1, 28, 28, 1))

datagen = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, 
            samplewise_std_normalization=True)

datagen = datagen.flow(img, batch_size=1)

x = next(datagen)
print(x.shape)

model = keras.models.load_model('mnist.h5')
y = model.predict(x).reshape(10,)
print(y, np.argmax(y))