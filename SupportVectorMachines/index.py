
"##  Support Vector Machines"

# "Penjelasan: Random forest adalah ensemble learning method yang menggabungkan beberapa decision trees untuk meningkatkan akurasi dan mengurangi overfitting.
#              Setiap tree dihasilkan dari subset data yang berbeda dan hasil akhirnya adalah voting atau rata-rata dari semua tree."
# "Tujuan : SVM bertujuan untuk menemukan hyperplane terbaik yang memisahkan dua kelas data, hyperplane ini adalah batas keputusan yang memaksimalkan margin(jarak) antar dua kelas."
# "Penggunaan : Digunakan untuk klasifikasi biner, multi-kelas, dan regresi, seperti pengenalan wajah, klasifikasi teks, dan bioinformatika."

# Konsep Dasar Support Vector Machines
# 1.Objectif Utama :
#    - SVM bertujuan untuk menemukan hyperplane terbaik yang memisahkan dua kelas data.
#    - Hyperplane ini adalah batas keputusan yang memaksimalkan margin (jarak) antara dua kelas.
# 2.Hyperplane dan Margin:
#    - Hyperplane dalam SVM adalah generalisasi dari konsep garis pemisah dalam dimensi yang lebih tinggi.
#    - Margin adalah jarak antara hyperplane dan instance terdekat dari setiap kelas. SVM berusaha untuk memaksimalkan margin ini.
# 3. Support Vectors:
#    - Support vectors adalah titik-titik data yang berada paling dekat dengan hyperplane. Mereka adalah titik-titik yang paling mempengaruhi posisi dan orientasi hyperplane.
#    - Hanya support vectors yang berkontribusi pada pembentukan hyperplane, sehingga SVM efisien dalam penggunaan memori.
# 4. Kernel Trick:
#    - SVM dapat melakukan transformasi non-linear menggunakan fungsi kernel, yang memungkinkan penanganan masalah klasifikasi yang tidak dapat dipisahkan secara linear di ruang dimensi yang lebih tinggi.
#    - Kernel seperti Polynomial, Gaussian Radial Basis Function (RBF), dan Sigmoid digunakan untuk memperluas representasi data ke dimensi yang lebih tinggi.
# 5. Regulasi:
#    - SVM juga menggunakan regularisasi untuk mengontrol kompleksitas model dan mencegah overfitting.
#    - Parameter C digunakan untuk mengatur tingkat regularisasi dalam SVM, dengan nilai yang lebih tinggi menunjukkan regularisasi yang lebih lemah.

# Rumus SVM Support Vector Machines
# SVM memiliki berbagai macam rumus untuk sesuai dengan fungsional
# 1. SVM untuk Klasifikasi Biner dengan Hyperplane Linear
#     Misalkan kita memiliki dua kelas yang ingin dipisahkan, yaitu kelas positif (+1) dan kelas negatif (-1). SVM mencari hyperplane w * x + b = 0 yang memisahkan kedua kelas dengan margin maksimum.
#     - w adalah vektor normal ke hyperplane.
#     - x adalah vektor fitur dari instance data.
#     - b adalah bias
#     Untuk prediksi, SVM menggunakan fungsi sign dari nilai  : w * x + b
#     y^ = sign(w * x + b)
#     Dimana y^ adalah label prediksi (+1 atau -1)

# 2. Mengoptimalkan Margin
#     SVM mencari hyperplane dengan margin maksimum. Margin adalah jarak antara hyperplane dan support vectors terdekat. Pemaksimalan margin ini dapat diformulasikan sebagai masalah optimasi dengan menggunakan fungsi objektif:
#       Minimize 1/2||w||^2
#     dibawah kendala
#     yi(w * Xi +b) ≥ 1 untuk semua i = 1, ..., N
#     dimana yi, adalah label kelas dari instance xi dan N adalah jumlah instance data

# 3. Kernel SVM
#     SVM juga dapat menangani data yang tidak dapat dipisahkan secara linear dengan menggunakan kernel. Kernel memungkinkan SVM untuk melakukan transformasi non-linear data ke ruang fitur yang lebih tinggi. Contoh kernel yang umum digunakan termasuk:
#     -  Linear Kernel : K(Xi, Xj) = xi * xj
#     -  Polynomial Kernel : K(Xi, Xj) = (yxi * xj + r)^d
#     -  Gaussian Radial Bassis Function (RBF) Kernel : K(xi, xj) = exp(-y||xi - xj||^2)
#     -  Sigmoid Kernel : K(Xi, Xj) = tanh(yXi * Xj + r)
#       dimana γ, 𝑟 dan 𝑑 adalah parameter kernel yang dapat disesuaikan.


# Implementasi SVM

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# memuat data
df = pd.read_csv('../0_Data/Iris.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# menampilkan data
print(df.head())
print(df.columns)


typeDatas = df.dtypes.value_counts()
print(typeDatas)

# Melakukan check apakah didalam data ada column yang memiliki nilai null/missing value
theresNull = df.isna().sum()
print(theresNull)

df = df.drop(columns=['Id'])

# Memilih fitur (X) dan target (y)
X = df.drop(columns=['Species'])
y = df['Species']

print("X : ", X)
print("y : ", y)
#
# # menggunakan dua fitur pertama untuk klasifikasi biner
X = X.iloc[:, :2] # menggunakan iloc untuk memilih kolom dengan indeks
print("X : ", X)

# Mengganti label kelas menjadi biner (0 dan 1)
# y = (y == 'Iris-setosa').astype(int)

# Harus jadikan One Hot EndCoding jika ingin menggunakan multiple
label_encoder = LabelEncoder()
y_encoder = label_encoder.fit_transform(y)
print("y : ", y_encoder)

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoder, test_size=0.2, random_state=42)

# Inisialisasi model SVM dengan
model = SVC(kernel='linear', C=1.0, random_state=42)

# Training model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy}')
print(report)


# Visualisasi data dan decision boundary
plt.figure(figsize=(10, 6))

# Menampilkan data training
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='winter', marker='o', s=50, label='Training Data')

# Menampilkan data testing
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='cool', marker='x', s=50, label='Testing Data')

# Menghasilkan mesh grid untuk membuat decision boundary
h = .02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prediksi pada setiap titik dalam mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Menampilkan decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='winter')

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('SVM Decision Boundary and Data Points')
plt.legend()
plt.show()
