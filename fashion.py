import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
fashion_data = tf.keras.datasets.fashion_mnist.load_data()
(train_set, train_labels), (test_set, test_labels) = fashion_data

# Normalize and reshape images
train_set = train_set.reshape(-1, 28, 28, 1).astype('float32') / 255
test_set = test_set.reshape(-1, 28, 28, 1).astype('float32') / 255

# Define the CNN architecture
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

# Compile the model
cnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Display model summary
cnn_model.summary()

# Train the model
training_history = cnn_model.fit(
    train_set, train_labels, 
    epochs=10, 
    validation_split=0.2
)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(test_set, test_labels, verbose=2)
print(f'\nTest accuracy: {test_accuracy:.4f}')

# Make predictions on sample images
sample_images = test_set[:2]
predictions = cnn_model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)

# Define class labels
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Print predictions
for i, pred_class in enumerate(predicted_classes):
    print(f"Prediction for sample {i+1}: {fashion_classes[pred_class]}")

# Visualize predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(sample_images[0].reshape(28, 28), cmap='gray')
ax1.set_title(f"Predicted: {fashion_classes[predicted_classes[0]]}")
ax2.imshow(sample_images[1].reshape(28, 28), cmap='gray')
ax2.set_title(f"Predicted: {fashion_classes[predicted_classes[1]]}")
plt.show()
