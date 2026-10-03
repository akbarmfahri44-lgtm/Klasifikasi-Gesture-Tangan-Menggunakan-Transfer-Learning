import cv2
import tensorflow as tf
import numpy as np

# Memuat model
model = tf.keras.models.load_model("model/gesture_model.keras")

# Nama kelas
class_names = ["fist", "palm", "victory"]

# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

print()
print("========================================")
print("     TEST GESTURE DENGAN KAMERA")
print("========================================")
print("Tekan Q untuk keluar")
print("========================================")
print()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    # Mirror kamera
    frame = cv2.flip(frame, 1)

    # Resize gambar
    image = cv2.resize(frame, (224, 224))

    # BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ubah menjadi array
    image = np.array(image)

    # Tambahkan batch
    image = np.expand_dims(image, axis=0)

    # Preprocessing MobileNetV2
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Prediksi
    prediction = model.predict(image, verbose=0)

    # Kelas dengan nilai tertinggi
    class_index = np.argmax(prediction[0])

    confidence = prediction[0][class_index] * 100

    gesture = class_names[class_index]

    # Tampilkan hasil
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = Keluar",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    # Tombol Q
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program selesai.")