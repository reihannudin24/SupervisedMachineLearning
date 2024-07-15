
def LinearRegeression():
    print("##  Linear Regression")
    print("Penjelasan: Linear regression adalah metode statistik untuk memodelkan hubungan antara satu atau lebih variabel independen (prediktor) dan variabel dependen (target) dengan menggunakan garis lurus.")
    print("Tujuan: adalah untuk menemukan garis lurus (linear) yang paling sesuai dengan data sehingga kita dapat memprediksi nilai variabel dependen berdasarkan nilai variabel independen.")
    print("Penggunaan: Digunakan untuk masalah prediksi atau estimasi nilai kontinu, seperti harga rumah, penjualan, suhu, dan lain-lain.")

    print("\n")

    print("## Komponen Utama Linear Regression:")
    print("1.Variabel Dependen (Y): Variabel yang ingin kita prediksi atau jelaskan.")
    print("2.Variabel Independen (X): Variabel yang digunakan untuk membuat prediksi atau penjelasan.")
    print("3.Koefisien (β): Parameter yang menentukan seberapa besar pengaruh setiap variabel independen terhadap variabel dependen.")
    print("4.Intercept (α): Nilai awal dari variabel dependen ketika semua variabel independen bernilai nol.")
    print("5.Residuals (ε): Perbedaan antara nilai yang diamati dan nilai yang diprediksi oleh model.")

    print("## Rumus Linear Regression:")
    print("1.Untuk satu variabel independen:")
    print("Y=α+βX+ϵ")

    print("2.Untuk lebih dari satu variabel independen (multiple linear regression):")
    print("Y=α+β1X1+β2X2+...+βnXn+ϵ")


    print("=========================================== \n\n")

def LogisticRegression():
    print("Logistic Regression")
    print("Penjelasan: Logistic regression digunakan untuk memodelkan probabilitas dari kelas tertentu. Berbeda dengan linear regression, logistic regression menggunakan fungsi logit (sigmoid) untuk menghasilkan output dalam bentuk probabilitas antara 0 dan 1.")
    print("Penggunaan: Digunakan untuk masalah klasifikasi biner, seperti deteksi email spam, diagnosis penyakit, prediksi churn pelanggan, dan lain-lain.")
    print("=========================================== \n\n")


def DecisionTrees():
    print("Decision Trees")
    print("Penjelasan: Decision trees adalah algoritma yang menggunakan struktur pohon untuk membuat keputusan berdasarkan aturan if-then-else yang dihasilkan dari data pelatihan.")
    print("Penggunaan: Digunakan untuk masalah klasifikasi dan regresi, seperti klasifikasi spesies tanaman, prediksi kegagalan mesin, penentuan kredit risiko, dan lain-lain.")
    print("=========================================== \n\n")


def RandomForrest():
    print("Random Forrest")
    print("Penjelasan: Random forest adalah ensemble learning method yang menggabungkan beberapa decision trees untuk meningkatkan akurasi dan mengurangi overfitting. Setiap tree dihasilkan dari subset data yang berbeda dan hasil akhirnya adalah voting atau rata-rata dari semua tree.")
    print("Penggunaan: Digunakan untuk masalah klasifikasi dan regresi, seperti prediksi penjualan, deteksi anomali, analisis pasar, dan lain-lain.")
    print("=========================================== \n\n")


def KNearestNeighbors():
    print("K-Nearest Neighbors (KNN)")
    print("Penjelasan: KNN adalah algoritma non-parametrik yang menggunakan kesamaan untuk mengklasifikasikan atau memprediksi nilai berdasarkan data terdekat. Algoritma ini menyimpan semua contoh pelatihan dan mengklasifikasikan titik baru berdasarkan mayoritas label k-neighbor terdekatnya.")
    print("Penggunaan: Digunakan untuk klasifikasi dan regresi, seperti pengenalan pola, deteksi anomali, dan rekomendasi.")
    print("=========================================== \n\n")


def NaiveBayes():
    print("Naive Bayes")
    print("Penjelasan: Naive Bayes adalah algoritma klasifikasi probabilistik berdasarkan Teorema Bayes dengan asumsi independensi antara fitur. Meski asumsi independensinya sederhana, algoritma ini sering memberikan hasil yang baik.")
    print("Penggunaan: Digunakan untuk klasifikasi teks (spam filtering, analisis sentimen), diagnosis medis, dan deteksi anomali.")
    print("=========================================== \n\n")


def GradientBoostingMachines():
    print("Gradient Boosting Machines")
    print("Penjelasan: GBM adalah teknik ensemble yang membangun model prediktif dari sejumlah model lemah (biasanya decision trees) yang digabungkan dalam cara iteratif untuk mengurangi error secara bertahap.")
    print("Penggunaan: Digunakan untuk klasifikasi dan regresi dengan kinerja tinggi, seperti prediksi risiko kredit, penjualan produk, dan analisis data keuangan.")
    print("=========================================== \n\n")


def AdaBoost():
    print("Ada Boost")
    print("Penjelasan: AdaBoost (Adaptive Boosting) adalah algoritma ensemble yang menggabungkan beberapa model lemah untuk membentuk model kuat dengan memberikan bobot lebih pada instance yang salah diklasifikasikan pada iterasi sebelumnya.")
    print("Penggunaan: Digunakan untuk klasifikasi dan regresi, khususnya dalam pengenalan wajah dan deteksi objek.")
    print("=========================================== \n\n")


def NeuralNetwork():
    print("Neural Network")
    print("Penjelasan: Neural networks adalah model yang terinspirasi oleh jaringan neuron biologis, terdiri dari lapisan-lapisan neuron yang saling terhubung. Setiap neuron menerima input, memprosesnya, dan mengirim output ke neuron berikutnya.")
    print("Penggunaan: Digunakan untuk berbagai aplikasi, termasuk pengenalan gambar, pengolahan bahasa alami, prediksi waktu, klasifikasi, dan lain-lain.")
    print("=========================================== \n\n")

def BayesianNetworks():
    print("BayesianNetworks")
    print("Penjelasan: Bayesian Networks adalah model grafis probabilistik yang mewakili serangkaian variabel acak dan ketergantungan kondisionalnya melalui graf berarah.")
    print("Penggunaan: Digunakan untuk pengambilan keputusan, diagnosis medis, prediksi risiko, dan analisis sebab-akibat.")
    print("=========================================== \n\n")


def SupportVectorMachines():
    print("Support Vector Machines ")
    print("SVM adalah algoritma yang mencari hyperplane optimal yang memisahkan data dalam ruang fitur yang lebih tinggi. SVM berfokus pada margin yang maksimal antara kelas yang berbeda.")
    print("Digunakan untuk klasifikasi biner, multi-kelas, dan regresi, seperti pengenalan wajah, klasifikasi teks, dan bioinformatika.")
    print("=========================================== \n\n")


def SupportVectorRegression():
    print("Support Vector Regression")
    print("Penjelasan: SVR adalah varian dari Support Vector Machines (SVM) yang digunakan untuk masalah regresi. SVR berusaha menemukan hyperplane yang memaksimalkan margin antara data poin dengan mempertimbangkan toleransi error tertentu.")
    print("Penggunaan: Digunakan untuk masalah prediksi nilai kontinu, seperti prediksi harga saham, estimasi biaya, prakiraan cuaca, dan lain-lain.")
    print("=========================================== \n\n")


def LinearDiscriminantAnalysis():
    print("Linear Discriminant Analysis (LDA)")
    print("Penjelasan: LDA adalah metode yang digunakan untuk menemukan kombinasi fitur yang memisahkan dua atau lebih kelas. Ini digunakan untuk mengurangi dimensi dan untuk klasifikasi.")
    print("Penggunaan: Digunakan untuk pengenalan pola, deteksi penipuan, dan analisis data genetik.")
    print("=========================================== \n\n")


LinearRegeression()
LogisticRegression()
DecisionTrees()
RandomForrest()
KNearestNeighbors()
NaiveBayes()
GradientBoostingMachines()
AdaBoost()
NeuralNetwork()
BayesianNetworks()
SupportVectorMachines()
SupportVectorRegression()
LinearDiscriminantAnalysis()