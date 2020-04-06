from tensorflow import keras
import cv2

def inference(inputs):
    inputs = keras.layers.Flatten()(inputs)
    x1 = keras.layers.Dense(512,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x1 = keras.layers.Dense(256,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x1)
    output = keras.layers.Dense(10,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))(x1)
    return output

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,28, 28, 1)
x_test = x_test.reshape(10000,28, 28, 1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

datagen = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, 
            samplewise_std_normalization=True, 
            width_shift_range=0.125,
            height_shift_range=0.125,
            fill_mode='constant')
datagen.fit(x_train)

img_input = keras.layers.Input(shape=(28,28,1))
output = inference(img_input)
model = keras.models.Model(img_input, output)

sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
    epochs=5,steps_per_epoch= 60000//64 + 1, validation_data=(x_test,y_test))

model.save('mnist.h5')