"##  Logistic Regression"
# "Penjelasan: Logistic regression digunakan untuk memodelkan probabilitas dari kelas tertentu. Logistic regression  digunakan untuk memprediksi probabilitas biner (dua kelas),
#              bukan nilai kontinu seperti dalam regresi linier. Berbeda dengan linear regression, logistic regression menggunakan fungsi logit (sigmoid) untuk menghasilkan output dalam bentuk probabilitas antara 0 dan 1."
# "Tujuan : "8
# "Penggunaan: Digunakan untuk masalah klasifikasi biner, seperti deteksi email spam, diagnosis penyakit, prediksi churn pelanggan, dan lain-lain."


# Konsep dasar Logistic Regression
# 1. Fungsi Logistik (Sigmoid):
#    - Logistic Regression menggunakan fungsi logistik, juga dikenal sebagai sigmoid function, untuk memetakan input ke output yang berada dalam rentang [0, 1].
#    - Sigmoid function didefinisikan sebagai:
#                σ(z) = 1/1+e^-z
#       di mana 𝑧 adalah hasil dari fungsi linear yang menggabungkan fitur (𝑧 = β0 + β1x1 +  β2x2 + ... + βnxn)
# 2. Model Linear:
#     - Model Logistic Regression memprediksi probabilitas bahwa suatu instance (misalnya, data point) termasuk dalam kelas tertentu.
#     - Rumus matematisnya adalah:
#          P(Y=1∣X)=σ(β0 + β1x1 +  β2x2 + ... + βnxn)
#         di mana P(Y=1∣X) adalah probabilitas bahwa instance X termasuk dalam kelas positif dan σ adalah fungsi sigmoid.
# 3. Fitting Model:
#     - Proses fitting (pelatihan) Logistic Regression melibatkan menemukan koefisien β yang optimal dengan meminimalkan fungsi biaya (cost function) berdasarkan data pelatihan.
#     - Umumnya, fungsi biaya yang digunakan adalah log loss (cross-entropy loss) untuk klasifikasi biner.

# Implementasi Logistic Regression

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Memuat dataset
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

# memilih fitur (X) dan target (y)
X = df.drop(columns=['Species'])
y = df['Species']

# membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# inisialisai model logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)

# Training model
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy : {accuracy}")
print(f"Report : {report}")

