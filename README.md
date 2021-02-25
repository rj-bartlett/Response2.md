### Informal Response 2


##### One of the first steps towards teaching a computer "how to see" is splitting the data into two groups, a set of training images and training labels and then also a set of test images and test labels. Why is this done? What is the purpose of splitting the data into a training set and a test set?
Splitting data into a training set and a test set is important because it can test the accuracy of the machine learning algorithm without any unnecessary bias. Having the testing data be from the same dataset the computer learned makes sure that all images in the testing set will correspond to similar images and the same labels as the testing group. If the data was tested on the same data it was trained on, it would be similar to a student memorizing the answer key instead of taking similar practice tests. 

##### What is the purpose of the relu and softmax functions in the activation arguments in the Dense layers of our neural network?
1) Relu is "a piecewise linear function that will output the input directly if it is positive, otherwise it is 0." It is easier to train data on and it also achieves better performance than other activation arguments. 
``` python
if x > 0:
  return x
else:
  return 0
```
2) Softmax is a function that converts vectors into probabilities. These probabilities represent the relative scale for each unique value found in the layer. 

##### Why are there 10 neurons in the third and last layer in the neural network?
There are 10 neurons because there are 10 unique labels in the MNIST dataset.

##### How do the optimizer and loss functions operate to produce model parameters (estimates) within the model.compile() function?
Optimizers' role in the model.compile() function is to adjust weights and actively adapting the model to increase accuracy. The loss funciton is the guide to the optimizers by giving the optimizer a number which explains what direction the optimizer needs to adjust in the future to make the model more accurate.



- What is the shape of the images training set? 
          (60000, 28, 28) 
- What is the length of the labels training set?
          60,000
- What is the shape of the images test set?
          (10000, 28, 28)
          

``` python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs = 10, callbacks = callbacks)
model.evaluate(x_test, y_test)
prob_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
classification = prob_model.predict(x_test)
print(classification[5])
plt.hist(np.argmax(classification[5]), range(0,9,1), align = 'left')
plt.show()
print(y_test[5])
```
OUTPUT:
Reached 99% accuracy so cancelling training!
313/313 [==============================] - 1s 3ms/step - 
loss: 0.0791 
accuracy: 0.9770
[0.08533701 0.23196515 0.08533701 0.08533701 0.08533712 0.08533701
 0.08533701 0.08533834 0.08533737 0.08533701]


[histogram](https://github.com/rj-bartlett/Response2.md/issues/1#issue-816795732)

Test label = 1
