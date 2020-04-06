import tensorflow as tf

a = tf.keras.layers.Input(shape=(368, 368, 3))

conv1 = tf.keras.layers.Conv2D(64, 3, 1)(a)
conv2 = tf.keras.layers.Conv2D(64, 3, 1)(conv1)
maxpool = tf.keras.layers.MaxPooling2D(pool_size=8, strides=8, padding='same')(conv2)
conv3 = tf.keras.layers.Conv2D(5, 1, 1)(maxpool)

inputs = a
outputs = conv3
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.mean_squared_error)

import numpy as np
data = np.random.rand(10, 368, 368, 3)
label  = np.random.rand(10, 46, 46, 5)

dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(10).repeat()
model.fit(dataset, epochs=5, steps_per_epoch=30)