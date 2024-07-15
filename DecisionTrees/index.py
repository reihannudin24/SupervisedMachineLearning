"##  Decision Trees"
# "Penjelasan:  Decision trees adalah algoritma yang menggunakan struktur pohon untuk membuat keputusan berdasarkan aturan if-then-else yang dihasilkan dari data pelatihan.
# "Tujuan : adalah unutk membuat model yang prediktif yang mengelompokan data berdasarkan interpretabilitas."
# "Penggunaan: Digunakan untuk masalah klasifikasi dan regresi, seperti klasifikasi spesies tanaman, prediksi kegagalan mesin, penentuan kredit risiko, dan lain-lain."

# "## Komponen utama decision trees "
# "1.Node : merupakan bagian utama dari decision tree yang merepresentasikan titik pemisahan. dan setiap node mewakili suatu fitur dari data"
# "2.Edge : merupakan hubungan atau garis yang menghubungkan satu node dengan node yang lainnya"
# "3.Root Node : merupakaan Node awal di Decision Tree"
# "4.Internal Node : merupakan node yang memiliki child/node2 lain dibawahnya"
# "5.Left Node : merupakan node di ujung cabang/terakhir Decision Tree yang tidak memiliki child/turunan"
# "6.Branch/Split : merupakan garis atau proses pemisahan data berdasarkan nilai fitur pada setiap node"
# "7.Criteria for Splitting : merupakan aturan yang digunakan untuk membagi dataset disetiap node"
# "8.Decision Rule : merupakan aturan keputusan yang digunakan untuk membuat prediksi pada setap leaf node"
# "9.Pruning : merupakan proses menghapus cabang cabang yang tidak signifikan"


# ## Konsep solusi dalam decision trees
# 1. Cara Memilih Fitur Pembagi (Splitting Criteria):
#    - Gini impurity (untuk klasifikasi): Digunakan untuk mengukur seberapa murni atau homogen sebuah node dalam hal kelas targetnya.
#    - Information Gain (untuk klasifikasi): Digunakan untuk mengukur penurunan ketidakpastian atau entropy setelah memilih suatu fitur untuk membagi dataset.
#    - Mean Squared Error (MSE) (untuk regresi): Digunakan untuk mengukur seberapa baik suatu fitur memisahkan data berdasarkan nilai target yang kontinu.
# 2. Cara Membuat Keputusan (Decision Rule):
#    - Untuk setiap leaf node, keputusan bisa berupa kelas mayoritas (untuk klasifikasi) atau rata-rata nilai target (untuk regresi) dari subset data yang diberikan.
# 3. Cara Menangani Overfitting (Pruning):
#    - Pruning adalah proses penting untuk mengurangi kompleksitas model dan meningkatkan kemampuan generalisasi dengan menghapus cabang-cabang yang tidak signifikan dari pohon keputusan.


# Contoh Penggunaan Decision Trees untuk regresi :
# misal kita akan menggunakan decision trees untuk memprediksi harga rumah (MEDV) bedasarkan fitur-fitur lainnya dari dataset


# mengimport semua library yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# memuat data set dari file csv
df = pd.read_csv('../0_Data/HousingData.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# menampilkan data
print(df.head())
print(df.columns)

# mendapatkan informasi tentang tipedata
typeDatas = df.dtypes.value_counts()
print(typeDatas)

# melakukan check apakah terdapat null didalam datanya
theresNull = df.isna().sum()
print(theresNull)

# mengatasi missing values, dengan cara menambahkan rata-rata dari nilai kolom yang kosong
df.fillna(df.mean(), inplace=True)
print(df.isna().sum()) # melakukan check lagi jika data sudah tidak ada yang nulll

# menyiapkan fitur (x) dan target (y)
X = df.drop(columns=['MEDV']) # menghapus kolom target dari fitur
y = df['MEDV'] # mengambil koloms target

# membagi data menjadi taining dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# inisialisasikan model decision tree regressor
model = DecisionTreeRegressor(random_state=42)

# Kenapa kita memakai ini DecisionTreeRegressor
# Masalah yang Diselesaikan: Digunakan untuk masalah regresi, yaitu ketika target yang diprediksi adalah variabel kontinu (misalnya, harga rumah, suhu, atau berat).
# Prediksi: Menghasilkan nilai numerik yang kontinu.
# Kriteria Pemisahan: Biasanya menggunakan Mean Squared Error (MSE) atau Mean Absolute Error (MAE) sebagai kriteria pemisahan untuk meminimalkan kesalahan prediksi pada data training.


# Kapan kita harus memakai ini DecisionTreeClassifier
# Masalah yang Diselesaikan: Digunakan untuk masalah klasifikasi, yaitu ketika target yang diprediksi adalah variabel kategori (misalnya, jenis kelamin, kelas bunga, atau status penyakit).
# Prediksi: Menghasilkan kelas atau label yang diskrit.
# Kriteria Pemisahan: Biasanya menggunakan Gini impurity atau Information Gain (entropy) sebagai kriteria pemisahan untuk memaksimalkan kejelasan pembagian kelas pada data training.

# Kesimpulan
# DecisionTreeRegressor: Memodelkan hubungan antara fitur dan target numerik kontinu.
# DecisionTreeRegressor: Menggunakan kriteria seperti Mean Squared Error (MSE) atau Mean Absolute Error (MAE) untuk meminimalkan kesalahan prediksi.
# DecisionTreeRegressor: Evaluasi model dilakukan menggunakan metrik seperti Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R-squared (R2) Score.
# Evaluation DecisionTreeRegressor:
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# DecisionTreeClassifier: Memodelkan hubungan antara fitur dan target kategori.
# DecisionTreeClassifier: Menggunakan kriteria seperti Gini impurity atau Information Gain (entropy) untuk memaksimalkan kejelasan pembagian kelas.
# DecisionTreeClassifier: Evaluasi model dilakukan menggunakan metrik seperti akurasi, precision, recall, F1-score, dan confusion matrix.
# Evaluation DecisionTreeClassifier:
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)


# Training model
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')

# Plot hasil prediksi
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Data aktual')
plt.ylabel('Prediksi')
plt.title('Prediksi dengan decision tree regressor')
plt.show()
