import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


 

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train/255.0, images_test/255.0

#reduce datasize
images_train = images_train[0:100]
labels_train = labels_train[0:100]

images_test = images_test[0:30]
labels_test = labels_test[0:30]

#print(labels_test[0])
#print(labels_test.shape)
#print(labels_test)

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

#print( prediction.shape )
#print( prediction[0].shape )
#print(prediction[0], labels_test[0])


def evaluate_prediction( prediction, truth ):
	#if prediction[truth] >= 0.5: 
	#	return True  
	if prediction[truth] == max(prediction):
		return True
	return False
def compute_accuracy(predictions, labels):
	Npred = float(predictions.shape[0])
	Ncorrect = 0.
	for i in range(int(Npred)):
		if(evaluate_prediction( predictions[i], labels[i] )):
			Ncorrect = Ncorrect + 1.

	return Ncorrect/Npred
			

print("calculating accuracy from prediction") 
acc = compute_accuracy(prediction, labels_test)
print("--categorical accuracy s.t. succesful categorization implies correct label takes max probability in prediction array--")
print("accuracy = ",acc)


print("calculating loss")
print("-- mean sparse categorical cross entropy --")
def xentropy(prediction, truth):
	#y=[0]*10
	#y[truth]=1.
	return np.log(prediction[truth])

#print(xentropy(prediction[0],labels_test[0]) )	
def compute_loss(predictions, labels):
	loss = np.empty([])
	for i in range(len(predictions)):
		#loss = loss + xentropy(predictions[i], labels[i])
		#loss = xentropy(predictions[i],labels[i])
		#print(loss)
		loss = np.append(loss, xentropy(predictions[i],labels[i] ) )

	#return loss
	#return np.average(loss)
	return np.mean(loss)
	#return np.median(loss)

loss = compute_loss(prediction, labels_test)
print("loss =", loss)



