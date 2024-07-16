
"##  NoiveBayes"

# Penjelasan : Bayesian Networks (BNs) adalah model grafis yang merepresentasikan probabilitas antara variabel-variabel.
#               Penggunaannya sangat luas, termasuk dalam diagnosis medis, pemrosesan bahasa alami, sistem pendukung keputusan dan banyak lagi
# Tujuan Penggunaan Bayesian Networks
# 1. Pemodelan Ketidakpastian: BNs memungkinkan pemodelan ketidakpastian dalam sistem kompleks dengan merepresentasikan
#    hubungan probabilistik antara variabel.
# 2. Inferensi Probabilistik: BNs digunakan untuk melakukan inferensi probabilistik, yaitu menghitung probabilitas
#    variabel tertentu berdasarkan bukti yang ada.
# 3. Pemahaman Hubungan: BNs membantu dalam memahami hubungan antara variabel dalam suatu sistem.
# 4. Prediksi: BNs dapat digunakan untuk memprediksi hasil berdasarkan bukti yang ada.

# Langkah-langkah Menggunakan Bayesian Networks
# 1. Definisikan Variabel : Tentukan variabel acak yang relevan untuk masalah yang ingin Anda selesaikan.
# 2. Struktur Jaringan : Tentukan struktur jaringan dengan membuat Directed Acyclic Graph (DAG) yang merepresentasikan
#                        ketergantungan kondisional antara variabel.
# 3. Spesifikasi Distribusi Probabilitas Kondisional (CPDs) : Untuk setiap node, tentukan distribusi probabilitas
#                        kondisional yang mengkuantifikasi efek dari parent nodes pada node tersebut.
# 4. Pelatihan Model : Jika memiliki data, gunakan metode estimasi parameter seperti Maximum Likelihood Estimation (MLE)
#                      atau Bayesian Estimation untuk menentukan CPDs dari data.
# 5. Inferensi : unakan metode inferensi untuk menghitung distribusi probabilitas posterior dari variabel-variabel yang
#                diminati, berdasarkan bukti yang ada.
# 6. Evaluasi dan Validasi Model : Evaluasi dan validasi model untuk memastikan bahwa model tersebut akurat dan dapat diandalkan.


# Contoh implementasi dalam python
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

#  Baca dataset
df = pd.read_csv('../0_Data/customer_churn_dataset.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Preprocessing (contoh konversi data kategorikal ke numerik)
# df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Pilih beberapa fitur untuk Bayesian Network
data = df[['Age', 'Gender', 'Tenure',
           'Payment Delay', 'Subscription Type', 'Contract Length',
           'Total Spend', 'Churn']]

data = pd.get_dummies(data , columns=['Subscription Type' , 'Contract Length' ])
data.columns = data.columns.str.replace(' ', '_')

#  membuat struktur jaringan
model = BayesianNetwork([
    ('Subscription_Type_Basic', 'Churn'),
    ('Subscription_Type_Premium', 'Churn'),
    ('Subscription_Type_Standard', 'Churn'),
    ('Contract_Length_Annual', 'Churn'),
    ('Contract_Length_Monthly', 'Churn'),
    ('Contract_Length_Quarterly', 'Churn'),
    ('Total_Spend', 'Churn'),
])

# Melatih model menggunakan estimasi maximum Likelihood
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Menambahkan CPDs (jika tidak menggunakan fit)
# Jika Anda tidak ingin menggunakan fit, Anda bisa menambahkan CPDs secara manual. Namun, dalam contoh ini, kita menggunakan fit.

# Inferensi
# Melakukan Inferensi
inference = VariableElimination(model)

# menghitung probabilitas churn untuk pelanggan dengan kontrak nbulanan dengan subsribtion type premium
query = inference.query(variables=['Churn'], evidence={'Subscription_Type_Premium' : 1, 'Contract_Length_Monthly' :1})
print(query)
