import numpy as np
import tensorflow as tf
print (tf.__version__)

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()

print images_train
print images_train.shape
print images_train.dtype

print images_test.shape

print labels_train.shape
print labels_train.dtype

images_train, images_test = images_train/255., images_test/255.


mlp = tf.keras.models.Sequential()
mlp.add( tf.keras.layers.Flatten( input_shape=(28,28)) )
mlp.add( tf.keras.layers.Dense(400, activation='relu') )
mlp.add( tf.keras.layers.Dense(400, activation='relu') )
mlp.add( tf.keras.layers.Dense(10, activation='softmax') )
