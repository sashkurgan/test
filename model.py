import pandas as pd
import numpy as np
import tensorflow


train_data = tensorflow.python.keras.preprocessing.image_dataset_from_directory(
    directory='dataset/train',
    image_size=(512,512),
    label_mode="binary",
    batch_size=32
)

# Create a test dataset
test_data = tensorflow.python.keras.preprocessing.image_dataset_from_directory(
    directory='dataset/validation',
    image_size=(512,512),
    label_mode="binary",
)
AUTOTUNE = tensorflow.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

model_1 = tensorflow.python.keras.models.Sequential([
  tensorflow.python.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  tensorflow.python.keras.layers.Conv2D(filters=10,
                         kernel_size=3, # can also be (3, 3)
                         activation="relu"),
  tensorflow.python.keras.layers.Conv2D(10, 3, activation="relu"),
  tensorflow.python.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tensorflow.python.keras.layers.Conv2D(10, 3, activation="relu"),
  tensorflow.python.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tensorflow.python.keras.layers.MaxPool2D(2),
  tensorflow.python.keras.layers.Flatten(),
  tensorflow.python.keras.layers.Dense(1, activation="sigmoid") # binary activation output
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
              optimizer=tensorflow.python.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=20,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))