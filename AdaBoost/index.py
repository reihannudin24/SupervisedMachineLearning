
"##  NoiveBayes"

# Penjelasan : AdaBoost (Adaptive Boosting) adalah teknik ensemble yang bertujuan untuk meningkatkan akurasi model
#              prediktif dengan menggabungkan beberapa model lemah (weak learners) menjadi satu model yang kuat.
#              Ini membantu dalam mengurangi kesalahan klasifikasi dan meningkatkan performa, terutama pada dataset yang kompleks atau tidak seimbang.

# Langkah-langkah Implementasi AdaBoost
# Inisialisasi Bobot: Setiap data point dalam dataset diberikan bobot yang sama pada awalnya. Bobot ini akan diperbarui di setiap iterasi.
# Pelatihan Model: Latih model pembelajaran yang lemah (misalnya, pohon keputusan kecil) pada dataset dengan bobot yang ditentukan.
# Evaluasi Kesalahan: Hitung kesalahan model pada data. Kesalahan dihitung sebagai proporsi kesalahan klasifikasi dari model yang dilatih.
# Perbarui Bobot: Perbarui bobot untuk setiap data point. Data yang salah klasifikasi akan mendapatkan bobot lebih besar, sehingga model selanjutnya akan lebih fokus pada data tersebut.
# Hitung Alpha: Hitung kontribusi model yang baru dilatih menggunakan rumus alpha, yang menunjukkan seberapa baik model tersebut (berdasarkan kesalahan).
# Gabungkan Model: Tambahkan model baru ke dalam ensemble, menggunakan bobot yang dihitung untuk memberikan kontribusi pada prediksi akhir.
# Ulangi: Ulangi langkah 2-6 untuk sejumlah iterasi yang telah ditentukan (misalnya, T model lemah) hingga model mencapai performa yang memuaskan.
# Prediksi Akhir: Untuk membuat prediksi akhir, semua model dalam ensemble digunakan, dan hasilnya dikombinasikan dengan menggunakan bobot alpha untuk menghasilkan prediksi akhir


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Membagi data menjadi train dan test
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model pembelajaran lemah (misalnya pohon keputusan)
base_estimator = DecisionTreeClassifier(max_depth=1)

ada_boost = AdaBoostClassifier(base_estimator, n_estimators=50, algorithm='SAMME')

# melatih model
ada_boost.fit(X_train, y_train)

# memprediksi
y_pred = ada_boost.predict(X_test)
# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.2f}")

#
# Kelebihan dan Kekurangan AdaBoost
# Kelebihan:
#
# Mampu meningkatkan akurasi model yang lemah.
# Bekerja baik dengan model yang sederhana (seperti pohon keputusan kecil).
# Mampu mengurangi overfitting jika digunakan dengan bijak.
# Kekurangan:
#
# Sensitif terhadap noise dan outlier karena fokus pada kesalahan klasifikasi.
# Membutuhkan banyak iterasi untuk mencapai hasil yang baik