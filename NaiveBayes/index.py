
"##  NoiveBayes"

# Penjelasan : Native Bayes adalah algoritma klasifikasi yang berdasarkan pada teorma Bayes dengan asumsi independensi
#              antar fitur. Meskipun asumsi ini jarang terpenuhi dalam kehidupan nyata.
# Tujuan : Naive Bayes digunakan untuk klasifikasi teks (seperti deteksi spam), analisis sentimen, pengenalan pola,
#          dan prediksi berbagai kategori berdasarkan data historis.

# Langkah-langkah menggunakan Naive Bayes dipython
# 1. Hitung Probabilitas Prior:
#    - Hitung probabilitas prior untuk setiap kelas dalam data pelatihan
#    - P(Ci) = Jumlah data dengan kelas Ci / Total jumlah data
# 2. Hitung Probbilitas Likelihood
#    - Hitung probabilitas likelihood untuk setiap fitur berdasarkan kelas.
#    - Untuk Gaussian Naive Bayes, asumsi distribusi normal digunakan untuk menghitung likelihood.
#    - Rumus Hitung Probabilitas Likelihood
#    - dimana  𝜇𝑖 adalah mean dan σi2 adalah variance dari fitur Xj untuk kelas 𝐶𝑖
# 3. Hitung Probabilitas Posterior
#    - Hitung probabilitas posterior untuk setiap kelas menggunakan teorema Bayes
#    - P(Ci|X) = P(X|Ci) P(Ci)/P(X)
#    - Dimana P(X) adalah probabilitas evidence yang dapat diabaikan untuk klasifikasi karena kita hanya perlu
#      membandingkan nilai posterior relatif
# 4. Preiksi Kleas
#    - Pilih kelas dengan probabilitas posteriror tertinggi sebagai predisi
#    - C^ = arg maxCi P(Ci|X)

# Implementasi Naive Bayes dalam python

# import libraby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# Load Dataset
df = pd.read_csv('../0_Data/Iris.csv')

# Konversi Target 'Species' ke numerik menggunakan Label Encode
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# drop column yang tidak diperlukan
df = df.drop(columns=['Id'])

# memilih fitur dan target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# X = df[['SepalLengthCm', 'SepalWidthCm']].values
y = df[['Species']].values

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

# inislisasi model Naive Bayes
model = GaussianNB()

# Training model
model.fit(X_train, y_train)


# Menghitung probabilitas prior (class prior probabilities)
print("Probabilitas prior (P(C_i)) : ", model.class_prior_)

# Menghitung mean (theta) dan variance (sigma^2) untuk setiap fitur dan kelas
print("Mean dari setiap fitur untuk setiap kelas (theta):", model.theta_)
print("Variance dari setiap fitur untuk setiap kelas (sigma^2):", model.var_)


# prediksi
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy : {accuracy}")
print(report)

# Visualisasi

# # Tentukan batas plot
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx , yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                           np.arange(y_min, y_max, 0.02))
#
# # predict class for setiap titik pada grid
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # change result predict back to shape grid
# Z = Z.reshape(xx.shape)
#
# # Plot result prediction
# plt.figure(figsize=(8, 6))
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
# # plot juga data latih dan uji
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolors='k', s=20 ,label='Training Data')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='s', s=30, label='Testing data')
# plt.xlim(xx.min(), xx.max())
# plt.xlim(yy.min(), yy.max())
# plt.title("Gaussian Naive Bayes Classifier")
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.legend()
# plt.show()



