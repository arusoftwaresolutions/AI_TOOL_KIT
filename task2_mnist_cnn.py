"""
MNIST CNN classifier using TensorFlow Keras
- Loads MNIST from tf.keras.datasets
- Builds a simple CNN, trains to >95% test accuracy
- Plots training/validation accuracy
- Shows predictions on 5 sample test images
Run (ensure TensorFlow installed):
    pip install -U tensorflow matplotlib seaborn
    python task2_mnist_cnn.py
Notes:
    On CPU this may take several minutes. In Colab, enable GPU for faster training.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks

# Reproducible
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
x_test = np.expand_dims(x_test, -1)

# One-hot labels
num_classes = 10
y_train_cat = utils.to_categorical(y_train, num_classes)
y_test_cat = utils.to_categorical(y_test, num_classes)

# Build CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Callbacks
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
# Train
history = model.fit(x_train, y_train_cat, epochs=12, batch_size=128, validation_split=0.1, callbacks=[es], verbose=2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}")

# Plot training/validation accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict on 5 random test images
import random
indices = random.sample(range(len(x_test)), 5)
preds = model.predict(x_test[indices])
for i, idx in enumerate(indices):
    pred_label = np.argmax(preds[i])
    conf = np.max(preds[i])
    true_label = y_test[idx]
    plt.figure(figsize=(2,2))
    plt.imshow(x_test[idx].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"True: {true_label} Pred: {pred_label} ({conf:.2f})")
    plt.show()

# Save model for deployment
model.save("mnist_cnn_model.h5")
print("Saved model to mnist_cnn_model.h5")