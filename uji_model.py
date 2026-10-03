import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("model/gesture_model.keras")

class_names = ["fist", "palm", "victory"]

dataset_path = "dataset"

correct = 0
total = 0

print("========================================")
print("       PENGUJIAN MODEL GESTURE")
print("========================================")
print()

for label in class_names:

    folder = os.path.join(dataset_path, label)

    print(f"Pengujian kelas: {label}")

    if not os.path.exists(folder):
        print("Folder tidak ditemukan!")
        continue

    class_correct = 0
    class_total = 0

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder, filename)

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )

        image = tf.keras.utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)

        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        prediction = model.predict(image, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        class_total += 1
        total += 1

        if predicted_class == label:
            class_correct += 1
            correct += 1

    if class_total > 0:
        accuracy = (class_correct / class_total) * 100

        print(
            f"Benar: {class_correct}/{class_total} "
            f"({accuracy:.2f}%)"
        )

    print()

overall_accuracy = (correct / total) * 100

print("========================================")
print("HASIL AKHIR")
print("========================================")
print(f"Total gambar : {total}")
print(f"Prediksi benar : {correct}")
print(f"Akurasi pengujian : {overall_accuracy:.2f}%")
print("========================================")