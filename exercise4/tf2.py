import tensorflow as tf
import tensorflow_datasets as tfds


il_train = tfds.load('mnist', split='train', shuffle_files=True).shuffle(60000).batch(32)


class MNISTmodel(tf.keras.Model):
  def __init__(self):
    super(MNISTmodel, self).__init__()
    self.layer0 = tf.keras.layers.Flatten(input_shape=(28,28))
    self.layer1 = tf.keras.layers.Dense(400, activation='relu')
    self.layer2 = tf.keras.layers.Dense(400, activation='relu')
    self.layer3 = tf.keras.layers.Dense(10, activation='softmax') 

  def call(self, X):
    X1 = self.layer0(X)
    X2 = self.layer1(X1)
    X3 = self.layer2(X2)
    Y = self.layer3(X3)
    return Y


mlp = MNISTmodel()


mlp_loss = tf.keras.metrics.Mean()
mlp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def iterate(images, labels):
  with tf.GradientTape() as tape:
    predictions = mlp(images)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
  gradients = tape.gradient(loss, mlp.trainable_variables)
  tf.keras.optimizers.Adam().apply_gradients(zip(gradients, mlp.trainable_variables))

  mlp_loss(loss)
  mlp_accuracy(labels, predictions)


for epoch in range(2):
  mlp_loss.reset_states()
  mlp_accuracy.reset_states()

  for example in il_train:
    iterate(example['image'], example['label'])

  print('Epoch {}/2 - loss: {} - accuracy: {}'.format(epoch+1, mlp_loss.result(), mlp_accuracy.result()))



print("test evaluation")
il_test = tfds.load('mnist', split='test', shuffle_files=False).batch(32)

mlp_acc_test = tf.keras.metrics.SparseCategoricalAccuracy()
#print(il_test)
#imgs = il_test[0]['image']
for example in il_test:
	image = example['image']

	test_predictions = mlp(image)
	#print(test_predictions)
	label = example['label']
	#print(label)
	
	mlp_acc_test(label,test_predictions)

	
print("accuracy:",mlp_acc_test.result())


