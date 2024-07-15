"##  Linear Discriminant Analysis"
# Penjelasan : adalah teknik statistik yang digunakan dalam machine learning dan statistik untuk klasifikasi dan pengurangan
#              dimensi, yang bertujuan untuk menemukan kombinasi linear dari fitur yang memaksimalkan pemisahan antar kelas
# Tujuan  : - Klasifikasi : Membedakan kelas yang berbeda dalam dataset berdasarkan fitur yang diberikan
#           - Pengurangan Dimensi : Mengurangi jumlah fitur dalam dataset sambil mempertahankan informasi yang penting untuk pemisahan kelas
# Penggunaan : Digunakan untuk pengenalan pola, deteksi penipuan, dan analisis data genetik.

# Prinsip Dasar LDA:
#  - LDA mencari kombinasi linear dari fitur yang memaksimalkan rasio antara varians antar kelas (between class variance) dan varians dLm kelas (within-class variance)
#  - LDA menghasilkan garis atau bidang yang memisahkan kelas-kelasa dalam dataset sehingga pemisahan antar kelas menjadi maksimal

# Langkah" LDA :
#  1. Menghitung Mean untuk setiap kelas
#  2. Menghitung Scatter Matrix dalam kelas
#  3. Menhitung Scatter Matrix antar kelas
#  4. Menghitung Matriks Transformasi

# Implementasi LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# Memuat datase
df = pd.read_csv('../0_Data/Iris.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None
              )
# menampilkan data
print(df.head())
print(df.columns)

# check type data
typeDatas =  df.dtypes.value_counts()
print(typeDatas)

# melakukan check apakah didalam data ada columns yang memiliki nilai null/missing value
theresNull = df.isna().sum()
print(theresNull)

# drop columns is
df = df.drop(columns=['Id'])

# Memilih fitur (X) dan target
X = df.drop(columns=['Species'])
y = df['Species']

print("X : ", X)
print("y : ", y)

# Membagi data menjadi triaining dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisasion and training model LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Evaluasi model
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")

# Define batas plot
x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Visualisasi
plt.figure(figsize=(8,6))


colors = ['red', 'green', 'blue']
target_names = df['Species'].unique()


for color, target_names in zip(colors, target_names):
    plt.scatter(X_train_lda[y_train == target_names, 0], X_train_lda[y_train == target_names, 1], alpha=0.8, color=color, label=target_names)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()


# Kesimpulan:
# - Linear Discriminant Analysis (LDA) adalah metode yang efektif untuk klasifikasi dan pengurangan dimensi.
# - LDA mencari kombinasi linear dari fitur yang memaksimalkan pemisahan antar kelas.
# - LDA dapat diterapkan menggunakan pustaka scikit-learn di Python dengan langkah-langkah yang jelas seperti yang ditunjukkan di atas.

