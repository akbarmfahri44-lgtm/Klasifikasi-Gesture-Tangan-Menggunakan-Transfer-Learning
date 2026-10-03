import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================================
# PENGATURAN DATASET
# ========================================

dataset_path = "dataset"

img_size = (224, 224)
batch_size = 16
epochs = 10

# ========================================
# MEMBACA DATASET
# ========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names

print()
print("========================================")
print("KELAS GESTURE")
print("========================================")
print(class_names)
print("========================================")
print()

# ========================================
# OPTIMASI DATASET
# ========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ========================================
# DATA AUGMENTATION
# ========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ========================================
# MODEL TRANSFER LEARNING
# MobileNetV2
# ========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Membekukan model dasar
base_model.trainable = False

# ========================================
# MEMBUAT MODEL
# ========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ========================================
# COMPILE MODEL
# ========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========================================
# MENAMPILKAN MODEL
# ========================================

model.summary()

# ========================================
# TRAINING
# ========================================

print()
print("========================================")
print("MEMULAI TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

# ========================================
# MEMBUAT FOLDER MODEL
# ========================================

os.makedirs("model", exist_ok=True)

# ========================================
# MENYIMPAN MODEL
# ========================================

model.save("model/gesture_model.keras")

print()
print("========================================")
print("TRAINING SELESAI")
print("========================================")
print("Model disimpan di:")
print("model/gesture_model.keras")
print("========================================")

# ========================================
# GRAFIK AKURASI
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training dan Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

os.makedirs("hasil", exist_ok=True)

plt.savefig(
    "hasil/accuracy.png"
)

plt.show()

# ========================================
# GRAFIK LOSS
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training dan Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "hasil/loss.png"
)

plt.show()

print()
print("Grafik disimpan di folder hasil.")