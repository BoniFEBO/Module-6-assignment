# Module 6 Assignment

## Overview

The script performs the following tasks:
1. Loads and preprocesses the Fashion MNIST dataset
2. Defines and compiles a CNN model
3. Trains the model on the dataset
4. Evaluates the model's performance
5. Makes predictions on sample images and visualizes the results


## Code Explanation

```python
fashion_data = tf.keras.datasets.fashion_mnist.load_data()
(train_set, train_labels), (test_set, test_labels) = fashion_data

train_set = train_set.reshape(-1, 28, 28, 1).astype('float32') / 255
test_set = test_set.reshape(-1, 28, 28, 1).astype('float32') / 255
```

This section loads the Fashion MNIST dataset, reshapes the images to include a channel dimension, and normalizes the pixel values to be between 0 and 1.

```
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

This defines a CNN with three convolutional layers, two max pooling layers, and two dense layers. The final layer has 10 units corresponding to the 10 classes in the Fashion MNIST dataset.

```
training_history = cnn_model.fit(
    train_set, train_labels, 
    epochs=10, 
    validation_split=0.2
)
```

The model is trained for 10 epochs, using 20% of the training data for validation.

```
test_loss, test_accuracy = cnn_model.evaluate(test_set, test_labels, verbose=2)
sample_images = test_set[:2]
predictions = cnn_model.predict(sample_images)
```
The model is evaluated on the test set, and predictions are made on two sample images.

```
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(sample_images[0].reshape(28, 28), cmap='gray')
ax1.set_title(f"Predicted: {fashion_classes[predicted_classes[0]]}")
ax2.imshow(sample_images[1].reshape(28, 28), cmap='gray')
ax2.set_title(f"Predicted: {fashion_classes[predicted_classes[1]]}")
plt.show()
```

This code visualizes the sample images along with their predicted labels.
