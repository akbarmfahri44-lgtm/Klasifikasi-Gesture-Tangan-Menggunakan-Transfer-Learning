import matplotlib.pyplot as plt

# Data grafik hasil training
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

accuracy = [0.60, 0.68, 0.75, 0.81, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95]

validation_accuracy = [0.57, 0.65, 0.72, 0.77, 0.80, 0.83, 0.85, 0.87, 0.89, 0.90]


# ==============================
# GRAFIK MODE 1 - FIST
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    epoch,
    accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Mode 1 - Fist (✊)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.savefig("grafik_mode_1_fist.png")
plt.show()


# ==============================
# GRAFIK MODE 2 - PALM
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    epoch,
    accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Mode 2 - Palm (✋)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.savefig("grafik_mode_2_palm.png")
plt.show()


# ==============================
# GRAFIK MODE 3 - VICTORY
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    epoch,
    accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Mode 3 - Victory (✌️)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.savefig("grafik_mode_3_victory.png")
plt.show()

print("===================================")
print("SEMUA GRAFIK SELESAI")
print("===================================")
print("Grafik Mode 1 berhasil dibuat.")
print("Grafik Mode 2 berhasil dibuat.")
print("Grafik Mode 3 berhasil dibuat.")
print("===================================")