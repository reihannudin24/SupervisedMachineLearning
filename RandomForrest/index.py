
"##  Random Forrest"
# "Penjelasan: Random forest adalah ensemble learning method yang menggabungkan beberapa decision trees untuk meningkatkan akurasi dan mengurangi overfitting.
#              Setiap tree dihasilkan dari subset data yang berbeda dan hasil akhirnya adalah voting atau rata-rata dari semua tree."
# "Tujuan : untuk Mengingkatkan akurasi prediksi dan mengurangi overfitting"
# "Penggunaan: Digunakan untuk masalah klasifikasi dan regresi, seperti prediksi penjualan, deteksi anomali, analisis pasar, dan lain-lain."


# "## Langkah Langka Random Forrest"
# 1.Bootstrap Sampling : Dari dataset asli dengan 𝑁 contoh, buat beberapa subset acak dengan penggantian (bootstrap samples).
#                        Setiap subset akan memiliki ukuran yang sama dengan dataset asli tetapi beberapa contoh mungkin terduplikasi sementara yang lain mungkin tidak ada.
# 2.Training Decision Trees: Untuk setiap bootstrap sample, latih decision tree. Namun, setiap kali descision tree melakukan pemisahan (split), pilih subset acak dari fitur untuk menentukan split terbaik
# 3.Aggreagating Result : - Untuk prediksi klasifikasi: Lakukan voting mayoritas di mana setiap pohon memberikan satu suara untuk kelas prediksi, dan kelas dengan suara terbanyak menjadi prediksi akhir.
#                         - Untuk prediksi regresi: Hitung rata-rata dari semua prediksi pohon untuk mendapatkan prediksi akhir.

# Contoh penggunaan RandomForrest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# memuat dataset boston
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

# Menyiapkan fitur (X) dan target (y)
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Insialisasi model random ForrestRegression
model = RandomForestRegressor(n_estimators=100, random_state=42)

# if use Forest Classifier
# Inisialisasi model Random Forest Classifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training Mode
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluasi mode if use Forest Classifier
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')


