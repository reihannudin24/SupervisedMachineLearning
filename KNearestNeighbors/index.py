
"##  K-Nearest Neighbors (KNN)"

# Penjelasan : KNN adalah algoritma non-parametrik yang menggunakan kesamaan untuk mengklasifikasikan atau memprediksi
#              nilai berdasarkan data terdekat. Algoritma ini menyimpan semua contoh pelatihan dan mengklasifikasikan titik baru berdasarkan mayoritas label k-neighbor terdekatnya.
# Penggunaan: Digunakan untuk klasifikasi dan regresi, seperti pengenalan pola, deteksi anomali, dan rekomendasi.

# Langkah-Langkah menggunakan KNN:
# 1. Pilih nilai k : Tentukan jumlah tetangga terdekat yang akan digunakan untuk prediksi
# 2. Hitung jarak : Ukur jarak antara data baru dan semua data dalam dataset
# 3. Indentifikasi tetangga terdekat : Pilih 'k' tetangga dengan jarak terdekat
# 4. Prediksi :
#     - Klasifikasi : Mayoritas label dari 'k' tetangga terdekat menentukan kelad data baru
#     - Regresi : nilai prediksi adalah rata - rata dari nilai 'k' tetangga terdekat

# Implementasi KNN dalam python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('../0_Data/Iris.csv')

# Periksa tipe data
print(df.dtypes)

# Jika 'Species' bukan numerik, konversi ke numerik jika memungkinkan
# Misalnya, jika 'Species' adalah string, Anda bisa menggunakan label encoding
# Contoh: df['Species'] = df['Species'].astype('category').cat.codes

# Drop kolom yang tidak diperlukan
df = df.drop(columns=['Id'])

# Memilih fitur dan target
X = df[['SepalLengthCm', 'SepalWidthCm']].values
y = df['Species']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Training model
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)

# Tentukan batas plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Prediksi kelas untuk setiap titik pada grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Ubah hasil prediksi kembali ke bentuk grid
Z = Z.reshape(xx.shape)

# Plot hasil prediksi
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


print("Z : ", Z)

# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
# # Plot juga data latih dan data uji
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20, label='Training data')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='s', s=30, label='Testing data')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("K-Nearest Neighbors (k=3)")
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.legend()
# plt.show()
