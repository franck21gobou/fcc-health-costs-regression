# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget -nc -q https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

# Convert categorical data to numbers
dataset['sex'] = (dataset['sex'] == 'male').astype(int)
dataset['smoker'] = (dataset['smoker'] == 'yes').astype(int)
dataset = pd.get_dummies(dataset, columns=['region'], prefix='', prefix_sep='', dtype=int)
dataset = dataset.astype('float32')
dataset.tail()

# Split: 80% train / 20% test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Pop off the "expenses" column to create the labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

print(len(train_dataset), 'train rows /', len(test_dataset), 'test rows')

# Build and train the model
tf.keras.utils.set_random_seed(0)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    keras.Input(shape=(train_dataset.shape[1],)),
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mae',
              metrics=['mae', 'mse'])
model.summary()

history = model.fit(
    train_dataset, train_labels,
    epochs=300,
    validation_split=0.2,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)]
)
print('Epochs trained:', len(history.history['loss']))
print('Final validation MAE: {:.2f}'.format(min(history.history['val_mae'])))

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)

