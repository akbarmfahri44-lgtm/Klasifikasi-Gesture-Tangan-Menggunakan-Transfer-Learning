import cv2
import tensorflow as tf
import numpy as np
import time

model = tf.keras.models.load_model("model/gesture_model.keras")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

print("========================================")
print("       PENGUKURAN LATENCY MODEL")
print("========================================")
print("Tekan Q untuk berhenti")
print()

latencies = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    start_time = time.perf_counter()

    prediction = model.predict(image, verbose=0)

    end_time = time.perf_counter()

    latency = (end_time - start_time) * 1000
    latencies.append(latency)

    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index] * 100

    class_names = ["fist", "palm", "victory"]
    gesture = class_names[class_index]

    cv2.putText(
        frame,
        f"Prediksi: {gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Latency: {latency:.2f} ms",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Latency Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(latencies) > 0:
    average_latency = np.mean(latencies)

    print()
    print("========================================")
    print("HASIL LATENCY")
    print("========================================")
    print(f"Jumlah pengujian : {len(latencies)}")
    print(f"Rata-rata latency : {average_latency:.2f} ms")
    print("========================================")