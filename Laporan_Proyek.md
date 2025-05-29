# Laporan Proyek Machine Learning - Salwa Sabira

## Domain Proyek

Kanker payudara merupakan salah satu penyebab utama kematian akibat kanker pada perempuan di seluruh dunia. Deteksi dini sangat penting untuk meningkatkan tingkat kelangsungan hidup dan efektivitas pengobatan. Dengan kemajuan teknologi dan ketersediaan data medis, penerapan machine learning dalam deteksi kanker payudara kini menjadi solusi yang efisien dan akurat. Breast Cancer Wisconsin Dataset memberikan data karakteristik sel tumor yang dapat digunakan untuk mengklasifikasikan tumor sebagai **jinak (benign)** atau **ganas (malignant)**.

Menurut World Health Organization (WHO), sekitar 2,3 juta perempuan didiagnosis dengan kanker payudara pada tahun 2020 \[1]. Oleh karena itu, model prediksi berbasis machine learning yang cepat dan akurat dapat menjadi alat bantu bagi tenaga medis dalam pengambilan keputusan klinis.

**Referensi:**
\[1] World Health Organization. (2021). *Breast cancer*. diambil dari [https://www.who.int/news-room/fact-sheets/detail/breast-cancer](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)


## Business Understanding

### Problem Statements
- Bagaimana cara mengklasifikasikan jenis tumor (jinak atau ganas) berdasarkan fitur-fitur karakteristik sel dari dataset Breast Cancer Wisconsin?
- Seberapa akurat model machine learning dalam mendeteksi kanker payudara berdasarkan data tersebut?

### Goals
- Mengembangkan model klasifikasi untuk memprediksi jenis tumor dengan akurasi tinggi.
- Menilai performa model menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

### Solution Statements
Dua pendekatan yang digunakan:

- **K-Nearest Neighbors (KNN)**: Algoritma berbasis kedekatan jarak antar data.
- **Logistic Regression (LogReg)**: Algoritma linier yang menghitung probabilitas klasifikasi biner.

Kedua model dievaluasi menggunakan metrik klasifikasi serta dilakukan **hyperparameter tuning** untuk meningkatkan performa model.

---

## Data Understanding

Dataset yang digunakan adalah **Breast Cancer Wisconsin Diagnostic Dataset** yang diambil dari:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

Dataset ini berisi fitur-fitur hasil pemeriksaan kanker payudara dari gambar digital aspirasi jarum halus (FNA).

- Jumlah data: 569 baris dan 33 kolom (termasuk ID dan diagnosis).
- Target: `diagnosis` (M = malignan, B = benign)
- Tidak ditemukan **missing values**.
- Ditemukan **1 duplikat**, dan telah dihapus pada tahap *Data Preparation*.

### Fitur Utama:
- `radius_mean`: Rata-rata jarak dari pusat ke titik batas.
- `texture_mean`: Variasi intensitas piksel.
- `perimeter_mean`, `area_mean`, `smoothness_mean`, dll.
- Fitur dengan akhiran `_se` = standard error.
- Fitur dengan akhiran `_worst` = nilai maksimum dari masing-masing fitur.

### Distribusi Target:
- **Benign**: 62%
- **Malignant**: 38%

Visualisasi awal dilakukan dengan histogram untuk masing-masing fitur guna melihat pola distribusi dan potensi perbedaan antara kelas.

---

## Data Preparation

Langkah-langkah preprocessing:

1. **Drop kolom tidak relevan**: `id`, `Unnamed: 32`.
2. **Menghapus duplikat**: Ditemukan 1 baris duplikat dan dihapus.
3. **Label Encoding**: `M` → 1 (malignan), `B` → 0 (benign).
4. **Split Data**: Data dibagi menjadi 80% data latih dan 20% data uji.
5. **Feature Scaling**: Standarisasi menggunakan `StandardScaler` pada fitur numerik.

> Catatan: Scaling dilakukan **setelah** data dibagi untuk menghindari *data leakage*.

---

## Modeling

### 1. K-Nearest Neighbors (KNN)

KNN adalah algoritma klasifikasi berbasis instance-based learning. Ia memprediksi kelas suatu data baru dengan:

- Menghitung jarak ke seluruh data latih.
- Memilih *k* tetangga terdekat.
- Menggunakan **mayoritas voting** untuk menentukan kelas.

**Hyperparameter penting:**
- `n_neighbors`: jumlah tetangga. Terlalu kecil → overfit, terlalu besar → underfit.
- `weights`: `'uniform'` (sama rata) vs `'distance'` (tetangga dekat lebih berpengaruh).
- `metric`: jenis jarak, misalnya `'euclidean'`, `'manhattan'`, `'minkowski'`.

### 2. Logistic Regression

Logistic Regression adalah model linier untuk klasifikasi biner. Ia menghitung probabilitas kelas positif menggunakan fungsi sigmoid:
  
$$
P(y=1|x) = \frac{1}{1 + e^{-z}} \quad \text{dengan} \quad z = w^Tx + b
$$

**Hyperparameter penting:**
- `C`: tingkat regularisasi. Semakin kecil → semakin kuat regularisasi (mencegah overfitting).
- `penalty`: jenis regularisasi (`'l2'` atau `'l1'`).
- `solver`: metode optimasi, seperti `'liblinear'` atau `'lbfgs'`.

### Hyperparameter Tuning

Dilakukan dengan `GridSearchCV` 5-fold cross-validation.

- **KNN**:
  - `n_neighbors`: [3, 5, 7, 9, 11]
  - `weights`: ['uniform', 'distance']
  - `metric`: ['euclidean', 'manhattan', 'minkowski']

- **Logistic Regression**:
  - `C`: [0.01, 0.1, 1, 10, 100]
  - `solver`: ['liblinear', 'lbfgs']

---

## Evaluation

### Metrik Evaluasi:
- **Accuracy**: proporsi prediksi benar dari seluruh data.
- **Precision**: proporsi positif yang benar-benar positif.
- **Recall**: proporsi data positif yang berhasil dikenali.
- **F1-Score**: rata-rata harmonik dari precision dan recall.

### Hasil Sebelum Tuning:

| Model              | Akurasi | Precision | Recall | F1-Score |
|--------------------|--------:|----------:|-------:|---------:|
| Logistic Regression| 97.37%  | 97.37%    | 97.37% | 97.37%   |
| KNN (default)      | 94.74%  | 94.74%     | 94.74%  | 94.74%    |

### Hasil Setelah Tuning:

| Model              | Akurasi | Precision | Recall | F1-Score |
|--------------------|--------:|----------:|-------:|---------:|
| Logistic Regression| **99.12%** | **99.31%** | **98.84%** | **99.06%** |
| KNN                | 96.49%  | 96.27%    | 96.27% | 96.27%   |

---

## Interpretasi Hasil

Model **Logistic Regression** menunjukkan performa yang lebih tinggi dibandingkan **KNN** di semua metrik evaluasi. Dengan akurasi sebesar **99.12%**, precision **99.31%**, recall **98.84%**, dan F1-score **99.06%**, model ini sangat efektif untuk klasifikasi tumor dengan risiko kesalahan yang sangat rendah.

Sementara itu, model **KNN** meskipun cukup baik, performanya berada di bawah Logistic Regression. Kekurangan utama KNN terletak pada sensitivitas terhadap outlier dan kompleksitas saat prediksi data baru.

---

## Model Terbaik

Berdasarkan hasil evaluasi, model terbaik untuk deteksi dini kanker payudara adalah **Logistic Regression** karena memberikan performa yang sangat baik, cepat, dan interpretatif. Model ini direkomendasikan untuk digunakan dalam aplikasi deteksi awal kanker payudara.
