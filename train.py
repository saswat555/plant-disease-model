import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import os

# Constants
IMG_SIZE = 456  
BATCH_SIZE = 16  
EPOCHS = 40  
FINE_TUNE_EPOCHS = 20  

# Load dataset
(ds_train, ds_test, ds_val), ds_info = tfds.load(
    "plant_village", split=["train[:80%]", "train[80%:95%]", "train[95%:]"], as_supervised=True, with_info=True
)

num_classes = ds_info.features["label"].num_classes  # Get number of classes

# Albumentations Augmentation Pipeline
# Define the augmentation pipeline
albumentations_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(),
    A.GaussNoise(),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])

# Albumentations Augmentation Function
def augment_image(image, label):
    def numpy_aug(img):
        img = img.astype(np.uint8)  # Ensure the image is in uint8 format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        augmented = albumentations_transform(image=img)["image"]
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        return augmented

    augmented = tf.numpy_function(numpy_aug, [image], tf.uint8)
    augmented = tf.image.convert_image_dtype(augmented, tf.float32)  # Normalize to [0,1]
    augmented = tf.reshape(augmented, [IMG_SIZE, IMG_SIZE, 3])  # Ensure shape

    return augmented, label

# Preprocessing Function (Resizing and Normalization)
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32)  
    return image, label

# Dataset Processing
ds_train = (ds_train
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x, y: augment_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .shuffle(1000)
            .prefetch(tf.data.AUTOTUNE))

ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model Function for Hyperparameter Tuning
def build_model(hp):
    base_model = EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(hp.Choice("dropout", [0.2, 0.3, 0.4]))(x)
    x = Dense(hp.Int("units", min_value=256, max_value=1024, step=256), activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3, 1e-4, 5e-5])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Hyperparameter Tuning
tuner = kt.Hyperband(
    build_model, objective="val_accuracy", max_epochs=10, factor=3, directory="tuner_results", project_name="plant_disease"
)
tuner.search(ds_train, validation_data=ds_val, epochs=10, callbacks=[EarlyStopping(monitor="val_loss", patience=3)])

# Train Best Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Initial Training
best_model.fit(
    ds_train, validation_data=ds_val, epochs=EPOCHS, 
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    ]
)

# Fine-Tuning (Unfreeze Last 50 Layers)
base_model.trainable = True
for layer in base_model.layers[:-50]:  
    layer.trainable = False  

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get("learning_rate") / 10),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Fine-Tuning
best_model.fit(
    ds_train, validation_data=ds_val, epochs=FINE_TUNE_EPOCHS,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

# Save Trained Model
model_path = "plant_disease_model"
best_model.save(model_path)
print(f"Model saved at {model_path}")

# Evaluate Model
test_loss, test_acc = best_model.evaluate(ds_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate Predictions for Confusion Matrix
y_true, y_pred = [], []
for images, labels in ds_test:
    predictions = best_model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true, y_pred = np.array(y_true), np.array(y_pred)
cm = confusion_matrix(y_true, y_pred)

# Get Class Names
try:
    class_names = ds_info.features["label"].names
except AttributeError:
    class_names = [str(i) for i in range(num_classes)]  # Fallback

# Confusion Matrix Visualization
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print Classification Report
print(classification_report(y_true, y_pred, target_names=class_names))
