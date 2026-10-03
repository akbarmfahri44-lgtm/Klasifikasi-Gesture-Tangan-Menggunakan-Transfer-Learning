import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/gesture_model.keras")

class_names = ["fist", "palm", "victory"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

print("========================================")
print("MODE 2 - PALM")
print("========================================")
print("Gesture target : Palm (✋)")
print("Tekan Q untuk keluar")
print("========================================")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    frame = cv2.flip(frame, 1)

    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index] * 100
    gesture = class_names[class_index]

    cv2.putText(
        frame,
        "MODE 2 - PALM",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Prediksi: {gesture}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Mode 2 - Palm", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program selesai.")