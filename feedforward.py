import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate synthetic data (example)
np.random.seed(0)
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, size=(100,))  # Binary labels (0 or 1)

# Define the neural network architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),  # 10 input features
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Example prediction
new_data = np.random.rand(1, 10)  # New data point with 10 features
prediction = model.predict(new_data)
print("Prediction:", prediction)



output:
Epoch 1/10
3/3 [==============================] - 1s 126ms/step - loss: 0.7083 - accuracy: 0.4500 - val_loss: 0.6923 - val_accuracy: 0.4500
Epoch 2/10
3/3 [==============================] - 0s 17ms/step - loss: 0.6930 - accuracy: 0.5125 - val_loss: 0.6987 - val_accuracy: 0.5000
Epoch 3/10
3/3 [==============================] - 0s 17ms/step - loss: 0.6853 - accuracy: 0.5750 - val_loss: 0.7068 - val_accuracy: 0.4500
Epoch 4/10
3/3 [==============================] - 0s 18ms/step - loss: 0.6813 - accuracy: 0.5750 - val_loss: 0.7168 - val_accuracy: 0.4000
Epoch 5/10
3/3 [==============================] - 0s 31ms/step - loss: 0.6801 - accuracy: 0.5375 - val_loss: 0.7268 - val_accuracy: 0.4500
Epoch 6/10
3/3 [==============================] - 0s 33ms/step - loss: 0.6780 - accuracy: 0.5375 - val_loss: 0.7343 - val_accuracy: 0.4500
Epoch 7/10
3/3 [==============================] - 0s 29ms/step - loss: 0.6760 - accuracy: 0.5375 - val_loss: 0.7373 - val_accuracy: 0.4500
Epoch 8/10
3/3 [==============================] - 0s 35ms/step - loss: 0.6745 - accuracy: 0.5375 - val_loss: 0.7399 - val_accuracy: 0.4500
Epoch 9/10
3/3 [==============================] - 0s 29ms/step - loss: 0.6719 - accuracy: 0.5500 - val_loss: 0.7391 - val_accuracy: 0.4000
Epoch 10/10
3/3 [==============================] - 0s 33ms/step - loss: 0.6704 - accuracy: 0.5500 - val_loss: 0.7365 - val_accuracy: 0.4000
1/1 [==============================] - 0s 124ms/step
Prediction: [[0.49368867]]
