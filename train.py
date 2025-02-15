import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import os

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Load dataset with specific split
splits = {
    "train": "train[:80%]", 
    "test": "train[80%:95%]", 
    "val": "train[95%:]"
}

(ds_train, ds_test, ds_val), ds_info = tfds.load(
    "plant_village", split=[splits["train"], splits["test"], splits["val"]], as_supervised=True, with_info=True
)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

# Prepare datasets
ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).shuffle(1000)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE)

# Hyperparameter tuning function
def build_model(hp):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(hp.Choice("dropout", [0.2, 0.3, 0.4]))(x)
    x = Dense(hp.Int("units", min_value=64, max_value=256, step=64), activation="relu")(x)
    output_layer = Dense(ds_info.features["label"].num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Hyperparameter tuning
tuner = kt.Hyperband(
    build_model, objective="val_accuracy", max_epochs=10, factor=3, directory="tuner_results", project_name="plant_disease"
)
tuner.search(ds_train, validation_data=ds_val, epochs=10, callbacks=[EarlyStopping(monitor="val_loss", patience=3)])

# Get best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

# Save trained model
model_path = "plant_disease_model.h5"
best_model.save(model_path)
print(f"Model saved at {model_path}")

# Evaluate model
test_loss, test_acc = best_model.evaluate(ds_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Generate predictions and ground truth labels
y_true = []
y_pred = []
confidences = []

for images, labels in ds_test:
    predictions = best_model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)
    confidences.extend(confidence_scores)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
confidences = np.array(confidences)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))

# Accuracy vs Confidence Score Plot
plt.figure(figsize=(10, 6))
plt.scatter(confidences, y_pred == y_true, alpha=0.5)
plt.xlabel("Confidence Score")
plt.ylabel("Accuracy (1 = Correct, 0 = Incorrect)")
plt.title("Accuracy vs Confidence Score")
plt.show()
