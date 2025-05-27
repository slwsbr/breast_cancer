# Laporan Proyek Machine Learning - Salwa Sabira

## 📌 Domain Proyek

Kanker payudara merupakan salah satu penyebab utama kematian akibat kanker pada perempuan di seluruh dunia. Deteksi dini sangat penting untuk meningkatkan tingkat kelangsungan hidup dan efektivitas pengobatan. Dengan kemajuan teknologi dan ketersediaan data medis, penerapan machine learning dalam deteksi kanker payudara kini menjadi solusi yang efisien dan akurat. Breast Cancer Wisconsin Dataset memberikan data karakteristik sel tumor yang dapat digunakan untuk mengklasifikasikan tumor sebagai **jinak (benign)** atau **ganas (malignant)**.

Menurut World Health Organization (WHO), sekitar 2,3 juta perempuan didiagnosis dengan kanker payudara pada tahun 2020 \[1]. Oleh karena itu, model prediksi berbasis machine learning yang cepat dan akurat dapat menjadi alat bantu bagi tenaga medis dalam pengambilan keputusan klinis.

**Referensi:**
\[1] World Health Organization. (2021). *Breast cancer*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/breast-cancer](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)

---

## 📌 Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan jenis tumor (jinak atau ganas) berdasarkan fitur-fitur karakteristik sel dari dataset Breast Cancer Wisconsin?
2. Seberapa akurat model machine learning dalam mendeteksi kanker payudara berdasarkan data tersebut?

### Goals

1. Mengembangkan model klasifikasi untuk memprediksi jenis tumor dengan akurasi tinggi.
2. Menilai performa model menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

### Solution Statements
Untuk mencapai tujuan tersebut, dua pendekatan digunakan:
1. **K-Nearest Neighbors (KNN):** Algoritma berbasis kedekatan jarak antar data.
2. **Logistic Regression (LogReg):** Algoritma linier yang menghitung probabilitas klasifikasi biner.
Kedua model akan diuji dan dievaluasi menggunakan metrik seperti akurasi, presisi, recall, dan F1-score. Hyperparameter tuning dilakukan untuk meningkatkan performa model.

---

## 📌 Data Understanding

Dataset yang digunakan berasal dari Kaggle: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download]

### **Jumlah dan Struktur Data**

Dataset terdiri dari 569 baris dan 33 kolom (termasuk ID dan diagnosis).

### **Fitur**

* **ID**: Identifikasi unik pasien (dihapus karena tidak relevan).
* **Diagnosis**: Target klasifikasi: M (malignant) atau B (benign).
* **30 fitur numerik** hasil ekstraksi dari gambar biopsi digital:

  * radius\_mean, texture\_mean, perimeter\_mean, area\_mean, dll.

### **Visualisasi Data dan EDA**

* Distribusi kelas target: sekitar 62% benign dan 38% malignant.
* Visualisasi distribusi tiap fitur dilakukan menggunakan histogram.

---

## 📌 Data Preparation

Langkah-langkah yang dilakukan:

1. **Drop kolom yang tidak relevan**, seperti `id` dan `Unnamed: 32`.
2. **Menghapus duplikat.**
3. **Label encoding** untuk variabel target: `M` → 1 (ganas), `B` → 0 (jinak).
4. **Feature scaling** menggunakan **StandardScaler** karena sebagian besar model ML sensitif terhadap skala data.
5. **Split data** menjadi 80% train dan 20% test set untuk evaluasi model.

📌 *Alasan:*
* Feature scaling penting untuk model seperti Logistic Regression agar konvergen dengan benar.
* Data split mencegah overfitting dan memungkinkan evaluasi performa model secara objektif.

---

## 📌 Modeling

### **1. K-Nearest Neighbors (KNN)**

* Model KNN digunakan dengan parameter default awal `n_neighbors=5`.
* Dilakukan tuning hyperparameter menggunakan GridSearchCV:

  * `n_neighbors`: \[3, 5, 7, 9, 11]
  * `weights`: \['uniform', 'distance']
  * `metric`: \['euclidean', 'manhattan', 'minkowski']

### **2. Logistic Regression (LogReg)**

* Digunakan model LogisticRegression dari `sklearn.linear_model`.
* Hyperparameter tuning dilakukan:

  * `C`: \[0.01, 0.1, 1, 10, 100] (regularisasi)
  * `solver`: \['liblinear', 'lbfgs']

### **Kelebihan dan Kekurangan**

* **KNN**:

  * (+) Mudah diimplementasikan, interpretasi intuitif.
  * (-) Lambat saat prediksi, sensitif terhadap fitur yang tidak diskalakan.
* **Logistic Regression**:

  * (+) Cepat, interpretatif, cocok untuk klasifikasi biner.
  * (-) Cenderung underfit pada data non-linear.

---

## 📌 Evaluation

Metrik evaluasi:

* **Akurasi**: persentase prediksi benar.
* **Precision**: proporsi positif yang diprediksi benar.
* **Recall**: proporsi kasus positif yang berhasil dideteksi (penting dalam medis!).
* **F1-Score**: harmoni antara precision dan recall.

### Hasil evaluasi (contoh):

| Model               | Akurasi  | Precision | Recall  | F1-Score  |
| ------------------- | -------  | --------- | ------  | --------  |
| Logistic Regression | 99.12%   | 99.31%    | 98.84%  | 99.06%    |
|KNN                  | 96.49%   | 96.27%    | 96.27%  | 96.27%    |

### **Model Terbaik**

Berdasarkan hasil evaluasi, **Logistic Regression** memiliki performa lebih tinggi dari KNN. Oleh karena itu, Logistic Regression dipilih sebagai model terbaik untuk digunakan dalam sistem deteksi dini kanker payudara.

