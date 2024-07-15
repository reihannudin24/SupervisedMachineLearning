"##  Linear Regression"
# "Penjelasan: Linear regression adalah metode statistik untuk memodelkan hubungan antara satu atau lebih variabel independen (prediktor) dan variabel dependen (target) dengan menggunakan garis lurus."
# "Tujuan: adalah untuk menemukan garis lurus (linear) yang paling sesuai dengan data sehingga kita dapat memprediksi nilai variabel dependen berdasarkan nilai variabel independen."
# "Penggunaan: Digunakan untuk masalah prediksi atau estimasi nilai kontinu, seperti harga rumah, penjualan, suhu, dan lain-lain."

# "## Komponen Utama Linear Regression "
# "1.Variabel Dependen (Y): Variabel yang ingin kita prediksi atau jelaskan."
# "2.Variabel Independen (X): Variabel yang digunakan untuk membuat prediksi atau penjelasan."
# "3.Koefisien (β): Parameter yang menentukan seberapa besar pengaruh setiap variabel independen terhadap variabel dependen."
# "4.Intercept (α): Nilai awal dari variabel dependen ketika semua variabel independen bernilai nol."
# "5.Residuals (ε): Perbedaan antara nilai yang diamati dan nilai yang diprediksi oleh model."

# "## Rumus Linear Regression:"
# "1.Untuk satu variabel independen:"
# "Y=α+βX+ϵ"

# "2.Untuk lebih dari satu variabel independen (multiple linear regression):"
# "Y=α+β1X1+β2X2+...+βnXn+ϵ"

#
# Contoh Penggunaan Linear Regression:
# Misalkan kita memiliki dataset yang berisi informasi tentang luas rumah (dalam meter persegi) dan harga rumah. Kita ingin memprediksi harga rumah berdasarkan luas rumah tersebut.
#
# Langkah-langkah:
# 1. Kumpulkan Data: Dapatkan dataset yang berisi variabel luas_rumah (X) dan harga_rumah (Y).
# 2. Plot Data: Visualisasikan data untuk melihat pola hubungan antara luas rumah dan harga rumah.
# 3. Hitung Koefisien: Gunakan metode least squares untuk menentukan nilai koefisien 𝛼 dan 𝛽.
# 4. Buat Model: Gunakan koefisien yang telah dihitung untuk membuat model prediksi.
# 5. Evaluasi Model: Periksa keakuratan model dengan mengukur residuals atau menggunakan metrics seperti R-squared.


# Contoh penggunaan


# Mengimport semua library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Memuat dataset dari file csv
df = pd.read_csv('../0_Data/HousingData.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Menampilkan data
print(df.head())
print(df.columns)

typeDatas = df.dtypes.value_counts()
print(typeDatas)

# Melakukan check apakah didalam data ada column yang memiliki nilai null/missing value
theresNull = df.isna().sum()
print(theresNull)

# mengatasi missing values, misalnya dengan menggantinya dengan rata rata kolom
df.fillna(df.mean(), inplace=True)
print(df.isna().sum()) # melakukan check lagi jika data sudah tidak ada yang null maka akan dilanjutkan

# Memilih fitur RM(avarage number of rooms per dwelling) dan target MEDV (harga rumah)
X = df[['RM']]
y = df[['MEDV']]

print("X : " , X)
print("y : " , y)

# Membagi data menjadi Training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train : ", X_train)
print("y_train : ", y_train)
print("X_test : ", X_test)
print("y_test : ", y_test)

# Inisialisasi model
model = LinearRegression()

# membuat model Training
model.fit(X_train, y_train)

# membuat pediksi antara X test (input test)
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot hasil
plt.scatter(X_test, y_test, color='blue', label='Data Aktual')
plt.plot(X_test, y_pred, color='red', label='Model Linear')
plt.xlabel('Rata-rata jumlah kamar per hunian (RM)')
plt.ylabel('Harga Rumah (MEDV)')
plt.legend()
plt.show()


# Menampilkan koefisien, intercept, MSE, dan R2 Score

# Kesimpulan
print(f'Intercept : {model.intercept_}')
# merupakan nilai awal dari variabel Y (dependen) ketika semua varibel X (independent) bernilai nol
# dalam kasus ini berarti harga rumah ketika kamar rata" adalah nol
# (meskipun secara praktis ini tidak masuk akal, tetapi secara matematis itu adalah titik awal garis regresi).

print(f'Koefisien (β): {model.coef_[0]}')
# merepresntasikan seberapa besar perubahan dalam variabel Y (dependen) untuk setiap satu unit perubahan dalam variabel X (Independen)
# Dalam kasus ini berarti berapa besar perubahan harga rumah untuk setiap penambbahan satu kamar rata-rata


print(f'Mean Squared Error (MSE): {mse}')
# MSE adalah rata-rata kuadrat dari perbedaan antara nilai yang diprediksi dan nilai yang sebenarnya,
# hal ini memberikan ukuran seberapa baik model memprediksi nilai targe

print(f'R2 Score: {r2}')
# R2 Score berkisar antara 0 dan 1. Nilai 1 berarti model sempurna dalam memprediksi nilai target,
# sedangkan nilai 0 berarti model tidak lebih baik daripada rata-rata sederhana dari nilai target.

