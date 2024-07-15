
"##  Support Vector Regression"

# "Penjelasan: SVR adalah varian dari Support Vector Machines (SVM) yang digunakan untuk masalah regresi. SVR berusaha menemukan hyperplane yang memaksimalkan margin antara data poin dengan mempertimbangkan toleransi error tertentu."
# "Tujuan : adalah meminimalkan kesalahan prediksi sambil menjaga model tetap sederhana."
# "Penggunaan: Digunakan untuk masalah prediksi nilai kontinu, seperti prediksi harga saham, estimasi biaya, prakiraan cuaca, dan lain-lain."

# Komponent Utama dalam SVR:
# 1. Kernel Functions:
#    - Linear
#    - Polynomial
#    - Radial Basis Function (RBF)
#    - Sigmoid
# 2. Epsilon (ε):
#    - Parameter yang menentukan margin di sekitar hyperplane di mana kesalahan tidak dihitung.
# 3. Regularization Parameter (C):
#    - Parameter yang mengontrol trade-off antara margin yang lebar dan kesalahan pada data pelatihan.


# Rumus Dalam SVR
# SVR mencoba meminimalkan fungsi berikut:
# 1/2 ||w||^2 + C ∑^n i = 1 max(0, yi - (w * xi + b) - ϵ)
# Dimana
# - w adalah vektor bobot.
# - C adalah parameter regularisasi.
# - ϵ adalah margin error.
# - yi adalah nilai target
# - xi adalah fitur input
# - b adalah bias

# Contoh implementasi SVR

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# memuat dataset
df = pd.read_csv('../0_Data/HousingData.csv')
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

# mengatasi missing values, misalnya dengan menggantinya dengan rata rata kolom
df.fillna(df.mean(), inplace=True)
print(df.isna().sum()) # melakukan check lagi jika data sudah tidak ada yang null maka akan dilanjutkan

# Memilih fitur (X) dan target (y)
X = df.drop(columns=['MEDV'])  # MEDV adalah nilai target (harga rumah)
y = df['MEDV']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train", X_train)
print("X_test", X_test)

# Inisialisasi model SVR dengan kernel RBF
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Training model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Y_pred" , y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')

# Visualisasi hasil prediksi vs nilai sebenarnya
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Support Vector Regression (SVR) Predictions')
plt.legend()
plt.show()



# =====================================================================
# RandomizedSearchCV akan mencoba n_iter (dalam hal ini 50) kombinasi acak dari parameter grid yang diberikan.
# kelebihan : Cepat untu proses
# Kekurangan : Tidak begitu akurat

# # Inisialisasi model SVR
# svr = SVR()
# # Menentukan parameter grid untuk RandomizedSearchCV
# param_dist = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': uniform(0.1, 100),
#     'gamma': uniform(0.001, 1),
#     'epsilon': uniform(0.1, 1)
# }
# # Inisialisasi RandomizedSearchCV
# random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=50, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)
# # Training model dengan RandomizedSearchCV
# random_search.fit(X_train, y_train)
# # Hasil parameter terbaik
# print("Best parameters found: ", random_search.best_params_)
# # Menggunakan model terbaik untuk prediksi
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)

# =====================================================================
# This is use GridSearchCv berarti akan menkombinasikan parameter SVR untuk mendapatkan hasil yang baik

# kelebihan : Sangat Akurat
# Kekurangan : Pemerosesan membutuhkan waktu yang cukup lama dan komputer yang sangat bagus

# My computer Cannot Compute :v
# # # Inisialisasi model SVR
# svr = SVR()
# # Menentukan parameter grid untuk GridSearchCV
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': [0.1, 1, 10, 100],
#     'gamma': [0.001, 0.01, 0.1, 1],
#     'epsilon': [0.1, 0.2, 0.5, 1]
# }
# # Inisialisasi GridSearchCV
# grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
# # Training model dengan GridSearchCV
# grid_search.fit(X_train, y_train)
# # Hasil parameter terbaik
# print("Best parameters found: ", grid_search.best_params_)
# # Menggunakan model terbaik untuk prediksi
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)