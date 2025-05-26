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

* Membangun model klasifikasi menggunakan dua algoritma berbeda: **Logistic Regression** dan **Random Forest Classifier**.
* Melakukan **hyperparameter tuning** pada model Random Forest untuk meningkatkan performa klasifikasi.
* Memilih model terbaik berdasarkan metrik evaluasi (F1-score dan recall — karena dalam konteks medis, false negative harus diminimalkan).

---

## 📌 Data Understanding

Dataset yang digunakan adalah **Breast Cancer Wisconsin Diagnostic Dataset**, yang tersedia di UCI Machine Learning Repository:
📥 [Link dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

Dataset ini terdiri dari 569 sampel data dan 30 fitur numerik hasil pemeriksaan mikroskopik sel tumor. Setiap sampel memiliki label diagnosis:

* `M` (Malignant – ganas)
* `B` (Benign – jinak)

### Variabel-variabel dalam dataset:

* `radius_mean`, `texture_mean`, `perimeter_mean`, ..., `fractal_dimension_worst`: fitur statistik dari bentuk dan tekstur sel.
* `diagnosis`: label target (M atau B).

---

## 📌 Data Preparation

Langkah-langkah yang dilakukan:

1. **Drop kolom yang tidak relevan**, seperti `id` dan `Unnamed: 32`.
2. **Label encoding** untuk variabel target: `M` → 1 (ganas), `B` → 0 (jinak).
3. **Feature scaling** menggunakan **StandardScaler** karena sebagian besar model ML sensitif terhadap skala data.
4. **Split data** menjadi 80% train dan 20% test set untuk evaluasi model.

📌 *Alasan:*

* Feature scaling penting untuk model seperti Logistic Regression agar konvergen dengan benar.
* Data split mencegah overfitting dan memungkinkan evaluasi performa model secara objektif.

---

## 📌 Modeling

Dua algoritma digunakan untuk membandingkan performa:

### 1. Logistic Regression

* Algoritma baseline yang sederhana dan cepat.
* Tidak memerlukan banyak tuning, cocok sebagai pembanding awal.

### 2. Random Forest Classifier

* Algoritma ensambel yang kuat dan mampu menangani non-linearitas.
* Dilakukan **Grid Search** untuk tuning parameter `n_estimators`, `max_depth`, dan `min_samples_split`.

📌 *Kelebihan:*

* Logistic Regression: mudah diinterpretasi.
* Random Forest: akurasi tinggi, mampu menangani fitur kompleks.

📌 *Kekurangan:*

* Logistic Regression: asumsi linieritas.
* Random Forest: interpretabilitas lebih rendah dibanding model linier.

---

## 📌 Evaluation

Metrik evaluasi:

* **Akurasi**: persentase prediksi benar.
* **Precision**: proporsi positif yang diprediksi benar.
* **Recall**: proporsi kasus positif yang berhasil dideteksi (penting dalam medis!).
* **F1-Score**: harmoni antara precision dan recall.

### Hasil evaluasi (contoh):

| Model               | Akurasi | Precision | Recall | F1-Score |
| ------------------- | ------- | --------- | ------ | -------- |
| Logistic Regression | 96.5%   | 95.2%     | 95.8%  | 95.5%    |
| Random Forest       | 98.2%   | 97.6%     | 98.4%  | 98.0%    |

📌 *Kesimpulan:*
Model **Random Forest** menunjukkan performa terbaik, terutama dalam recall yang penting untuk mencegah false negative dalam deteksi kanker.

