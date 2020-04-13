


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


 

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train/255.0, images_test/255.0

#reduce datasize
images_train = images_train[0:1000]
labels_train = labels_train[0:1000]


images_val = images_train[750:]
labels_val = labels_train[750:]

images_train = images_train[0:750]
labels_train = labels_train[0:750]

images_test = images_test[0:300]
labels_test = labels_test[0:300]



def createmodel(nNeuron, nHiddenLayer):
	mlp = tf.keras.models.Sequential()
	mlp.add( tf.keras.layers.Flatten(input_shape=(28, 28)) )
	for l in range(nHiddenLayer):
		mlp.add( tf.keras.layers.Dense(nNeuron, activation='relu') )
	
	mlp.add( tf.keras.layers.Dense(10, activation='softmax') )

	mlp.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

	return mlp

MAXEPOCHS = 5

mlp = createmodel(400,1)
result = mlp.fit(images_train,labels_train,MAXEPOCHS, validation_data=(images_val,labels_val))
print(mlp.metrics_names)
#print(result['loss'])
print(result.history['loss'])
#print("val")
print(result.history['val_loss'])
#mlp.fit(images_val,labels_val,1)
#print(labels_test[0])
#print(labels_test.shape)
#print(labels_test)


#test N neurons
Nneurons = np.linspace(100,1000,10)
Nepochs = list(range(MAXEPOCHS))
Nepochs = Nepochs[1:]
#print(Nepochs)
#print(Nneurons)



loss = np.empty( (len(Nepochs),len(Nneurons)) )
vloss = np.empty( (len(Nepochs),len(Nneurons)) )

#print(loss.shape)
#do neuron v loss
for i in range(len(Nepochs)):
	for j in range(len(Nneurons)):	
		print("training with epochs=",Nepochs[i]," neurons=",Nneurons[j])
		mlp = createmodel( Nneurons[j], 1 )
		result = mlp.fit(images_train, labels_train, Nepochs[i], validation_data=(images_val, labels_val))
		loss[i][j] = result.history['loss'][0]
		vloss[i][j] = result.history['val_loss'][0]


print(loss)
print(vloss)
		

"""
mlp = tf.keras.models.Sequential()
mlp.add( tf.keras.layers.Flatten(input_shape=(28, 28)) )
mlp.add( tf.keras.layers.Dense(400, activation='relu') )
mlp.add( tf.keras.layers.Dense(400, activation='relu') )
mlp.add( tf.keras.layers.Dense(10, activation='softmax') )


mlp.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
print("TF version",tf.__version__)

print("pre training evaluate")
mlp.evaluate(images_test, labels_test, verbose=2)

print("begin fit")
mlp.fit(images_train, labels_train, epochs=3)

batch_size=32
shuffle=True    # before each epoch
steps_per_epoch=None

print("post training evaluate")
mlp.evaluate(images_test,  labels_test, verbose=2)
print("calling predict")
prediction = mlp.predict(images_test)
"""
