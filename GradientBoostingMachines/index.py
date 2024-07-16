
"## Gradient Boosting Machines (GBM)"
# Pengenalan : GBM adalah metode ensemble yang membangun model yang kuat dengan menggabungkan prediksi dari beberapa
#              model yang lebih lemah. Berbeda dengan AdaBoost, yang menyesuaikan bobot data point berdasarkan kesalahan
#              klasifikasi, GBM secara iteratif mengurangi kesalahan residual dari model sebelumnya dengan menyesuaikan
#              model baru pada residual tersebut.
# Tujuan : utamanya adalah untuk meminimalkan fungsi kerugian (loss function).

# Langkah -langkah implementasi Gradient Boosting
# 1. Inisialisasi Model: Mulai dengan model awal, yang bisa menjadi model yang sangat sederhana seperti rata-rata dari target.
# 2. Hitung Residual: Hitung residual (kesalahan) antara prediksi model dan nilai target sebenarnya.
# 3. Pelatihan Model pada Residual: Latih model baru untuk memprediksi residual yang dihasilkan dari model sebelumnya.
# 4. Perbarui Prediksi: Tambahkan prediksi dari model baru ke prediksi model sebelumnya untuk mengurangi residual.
# 5. Ulangi: Ulangi langkah 2-4 untuk sejumlah iterasi yang telah ditentukan (misalnya, T model lemah) hingga model mencapai performa yang memuaskan.

# Implementasi Gradient Boosting dengan Python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting model
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# Parameter Penting
# n_estimators: Jumlah pohon keputusan atau iterasi boosting yang akan dilakukan.
# learning_rate: Faktor skala yang diterapkan pada setiap model lemah. Trade-off antara n_estimators dan learning_rate harus diperhatikan.
# max_depth: Kedalaman maksimum dari setiap pohon keputusan. Mencegah overfitting dengan mengendalikan ukuran pohon.


# Training the model
gbm.fit(X_train, y_train)

# Predict
y_pred = gbm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Kelebihan dan Kekurangan Gradient Boosting
# Kelebihan:
#
# Mampu menangani berbagai jenis data dan masalah prediksi.
# Mampu menangani fitur yang tidak seimbang.
# Sering kali menghasilkan akurasi yang tinggi.
# Kekurangan:
#
# Membutuhkan waktu pelatihan yang lebih lama dibandingkan metode ensemble lainnya.
# Rentan terhadap overfitting jika tidak ada regularisasi yang tepat.
# Membutuhkan penyetelan parameter yang cermat untuk performa optimal.


# Penyetelan Hyperparameter
# Untuk mengoptimalkan model GBM, kamu bisa menggunakan GridSearchCV atau RandomizedSearchCV dari scikit-learn untuk
# menemukan kombinasi parameter terbaik. Contoh penggunaan GridSearchCV:

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators' : [50, 100, 200],
    'learning_rate' : [0.01, 0.1, 0.2],
    'max_depth' : [3, 4, 5],
}

# Initialize Gradient Boosting model
gbm = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCv
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# predict with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")


