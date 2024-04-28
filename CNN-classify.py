import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models,layers,optimizers,preprocessing,Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

train_path = 'training_set'
test_path = 'test_set'

img_width, img_height = 128, 128
batch_size = 64
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator (
    rescale=1./255,
    rotation_range=20,      # Random rotation within the range [-20, 20] degrees
    width_shift_range=0.1,  # Random horizontal shift by up to 10% of the width
    height_shift_range=0.1, # Random vertical shift by up to 10% of the height
    shear_range=0.2,        # Shear intensity (shear angle in counter-clockwise direction in radians)
    zoom_range=0.2,         # Random zoom up to 20%
    horizontal_flip=True,   # Randomly flip inputs horizontally
    fill_mode='nearest'     # Strategy used for filling in newly created pixels
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Keep data in the same order as the directory
)

model = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Adding another Convolutional layer
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # Increasing the number of neurons in Dense layer
    tf.keras.layers.Dense(64, activation='relu'),   # Adding another Dense layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
)

test_generator.reset()  # Reset to start of dataset
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5).astype(int)

# Get true labels
y_true = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix with annotations
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Add text annotations
thresh = cm.max() / 2.  # Threshold for text color
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Cat', 'Dog'], rotation=45)
plt.yticks(tick_marks, ['Cat', 'Dog'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display classification report
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()