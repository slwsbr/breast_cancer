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

Kedua model dievaluasi menggunakan metrik klasifikasi serta dilakukan hyperparameter tuning untuk meningkatkan performa model.

---

## Data Understanding

Dataset yang digunakan adalah **Breast Cancer Wisconsin Diagnostic Dataset** yang diambil dari:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

Dataset ini juga dapat ditemukan di UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Dataset ini berisi fitur-fitur hasil pemeriksaan kanker payudara dari gambar digital aspirasi jarum halus (FNA).

- Jumlah data: 569 baris dan 33 kolom (termasuk ID dan diagnosis).
- Target: `diagnosis` (M = malignan, B = benign)
- Tidak ditemukan missing values.
- Ditemukan 1 duplikat, dan telah dihapus pada tahap *Data Preparation*.

### Uraian Seluruh Fitur:

* `id`: Nomor identifikasi pasien (tidak digunakan dalam analisis).
* `diagnosis`: Jenis diagnosis (M = Malignant, B = Benign), menjadi target prediksi.

**Fitur-fitur utama** dihasilkan dari tiga kelompok statistik: *mean* (rata-rata), *standard error (SE)*, dan *worst* (nilai maksimum). Fitur-fitur tersebut mencerminkan karakteristik sel kanker berdasarkan citra FNA.

#### Fitur-fitur dengan akhiran `_mean`:

* `radius_mean`: Rata-rata jarak dari pusat massa sel ke batasnya.
* `texture_mean`: Rata-rata variasi intensitas piksel dalam sel.
* `perimeter_mean`: Panjang rata-rata keliling kontur sel.
* `area_mean`: Luas rata-rata area sel.
* `smoothness_mean`: Ukuran kekhalusan batas sel, dihitung dari perubahan lokal pada panjang kontur.
* `compactness_mean`: Ukuran kekompakan sel, dihitung sebagai (perimeter² / area) - 1.
* `concavity_mean`: Derajat cekungan (lekukan) batas sel.
* `concave points_mean`: Jumlah titik-titik cekungan pada batas sel.
* `symmetry_mean`: Ukuran simetri sel.
* `fractal_dimension_mean`: Kompleksitas batas sel berdasarkan dimensi fraktal.

#### Fitur-fitur dengan akhiran `_se` (standard error):

* Mengukur variabilitas lokal dari masing-masing fitur di atas (misal: `radius_se`, `texture_se`, dll.).

#### Fitur-fitur dengan akhiran `_worst`:

* Menunjukkan nilai maksimum dari masing-masing fitur (misal: `radius_worst`, `area_worst`, `concavity_worst`, dll.).

### Distribusi Target:

* **Benign (jinak)**: 62%
* **Malignant (ganas)**: 38%

### Visualisasi Awal:

* Visualisasi awal dilakukan menggunakan histogram untuk setiap fitur, guna mengamati distribusi data serta perbedaan pola karakteristik antara tumor benign dan malignant.

---

## Data Preparation

Langkah-langkah preprocessing:

1. **Drop kolom tidak relevan**: `id`, `Unnamed: 32`.
2. **Menghapus duplikat**: Ditemukan 1 baris duplikat dan dihapus.
3. **Label Encoding**: `M` → 1 (malignan), `B` → 0 (benign).
4. **Split Data**: Data dibagi menjadi 80% data latih dan 20% data uji.
5. **Feature Scaling**: Standarisasi menggunakan `StandardScaler` pada fitur numerik.

> Catatan: Scaling dilakukan setelah data dibagi untuk menghindari *data leakage*.

---

## Modeling

Dua algoritma machine learning yang digunakan dalam proyek ini adalah **K-Nearest Neighbors (KNN)** dan **Logistic Regression**.

---

#### **1. K-Nearest Neighbors (KNN)**

KNN adalah algoritma klasifikasi berbasis instance-based learning yang tidak melakukan pelatihan eksplisit. Saat memprediksi kelas untuk suatu data baru, algoritma ini:

* Menghitung jarak antara data tersebut dengan seluruh data latih.
* Memilih *k* tetangga terdekat berdasarkan jarak tersebut.
* Menentukan kelas berdasarkan mayoritas kelas dari tetangga terdekat.

**Hyperparameter penting yang memengaruhi performa KNN:**

* `n_neighbors`: menentukan jumlah tetangga terdekat yang akan dipertimbangkan. Nilai terlalu kecil (misal 1 atau 3) bisa menyebabkan model terlalu sensitif terhadap noise (*overfitting*), sedangkan nilai terlalu besar bisa membuat prediksi terlalu umum (*underfitting*).

* `weights`: menentukan bobot kontribusi tetangga.

  * `'uniform'`: semua tetangga memiliki bobot yang sama.
  * `'distance'`: tetangga yang lebih dekat memiliki bobot lebih besar, sehingga pengaruhnya terhadap prediksi lebih besar. Biasanya memberikan hasil yang lebih baik, terutama jika data tersebar tidak merata.

* `metric`: jenis jarak yang digunakan untuk mengukur kedekatan, seperti:

  * `'euclidean'`: jarak lurus antar titik.
  * `'manhattan'`: jarak berdasarkan sumbu koordinat (grid-like).
  * `'minkowski'`: generalisasi dari Euclidean dan Manhattan.

Pemilihan metrik yang sesuai penting untuk mencerminkan struktur data yang sebenarnya.

---

#### **2. Logistic Regression**

Logistic Regression adalah model linier untuk klasifikasi biner yang memetakan input ke probabilitas kelas menggunakan fungsi **sigmoid**. Probabilitas ini kemudian digunakan untuk menentukan kelas akhir berdasarkan ambang batas (biasanya 0.5).

**Hyperparameter penting yang memengaruhi performa Logistic Regression:**

* `C`: parameter regularisasi yang mengontrol seberapa besar model "dihukum" karena kompleksitasnya.

  * Nilai **C kecil** → regularisasi kuat → mencegah overfitting, tetapi bisa mengurangi akurasi.
  * Nilai **C besar** → regularisasi lemah → model lebih bebas, tapi bisa overfit terhadap data latih.

* `penalty`: jenis regularisasi yang diterapkan:

  * `'l2'` (Ridge): mengurangi bobot besar secara merata, menjaga semua fitur tetap berkontribusi.
  * `'l1'` (Lasso): mendorong bobot tertentu menjadi nol, sehingga bisa digunakan untuk seleksi fitur otomatis.

Kombinasi `C` dan `penalty` sangat memengaruhi trade-off antara bias dan varians. Pemilihan nilai yang tepat membantu model menjadi lebih generalisasi dan tidak overfitting.


### Hyperparameter Tuning

Dilakukan dengan `GridSearchCV` 5-fold cross-validation.

- **KNN**:
  - `n_neighbors`: [3, 5, 7, 9, 11]
  - `weights`: ['uniform', 'distance']
  - `metric`: ['euclidean', 'manhattan', 'minkowski']

- **Logistic Regression**:
  - 'C': [0.01, 0.1, 1, 10]  
  - 'penalty': ['l1', 'l2']
  - 'solver': ['liblinear'] 

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
| Logistic Regression| 97.37%  | 97.37%    | 97.37% | 97.36%   |
| KNN (default)      | 94.74%  | 94.74%     | 94.74%  | 94.74%    |

### Hasil Setelah Tuning:

| Model              | Akurasi | Precision | Recall | F1-Score |
|--------------------|--------:|----------:|-------:|---------:|
| Logistic Regression| **99.12%** | **99.31%** | **98.84%** | **99.06%** |
| KNN                | 96.49%  | 96.27%    | 96.27% | 96.27%   |

---

## Interpretasi Hasil

Model **Logistic Regression** menunjukkan performa yang lebih tinggi dibandingkan KNN di semua metrik evaluasi. Dengan akurasi sebesar **99.12%**, precision **99.31%**, recall **98.84%**, dan F1-score **99.06%**, model ini sangat efektif untuk klasifikasi tumor dengan risiko kesalahan yang sangat rendah.

Sementara itu, model KNN meskipun cukup baik, performanya berada di bawah Logistic Regression. Kekurangan utama KNN terletak pada sensitivitas terhadap outlier dan kompleksitas saat prediksi data baru.

---

## Model Terbaik

Berdasarkan hasil evaluasi, model terbaik untuk deteksi dini kanker payudara adalah **Logistic Regression** karena memberikan performa yang sangat baik, cepat, dan interpretatif. Model ini direkomendasikan untuk digunakan dalam aplikasi deteksi awal kanker payudara.
