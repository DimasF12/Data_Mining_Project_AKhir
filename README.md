
# 💼 AML Fraud Detection – Logistic Regression Approach

Proyek ini bertujuan mendeteksi aktivitas transaksi mencurigakan yang dapat menjadi indikasi pencucian uang (*Anti Money Laundering / AML*) menggunakan algoritma *Logistic Regression*. Dataset yang digunakan bersumber dari Kaggle, yang berisi data transaksi keuangan dalam jumlah besar.

---

## 📌 Deskripsi Proyek

Dalam sistem keuangan modern, pencucian uang menjadi tantangan serius. Dengan pendekatan data mining, proyek ini:
- Mengklasifikasikan apakah suatu transaksi berpotensi fraud (pencucian uang).
- Menggunakan metode supervised learning (Logistic Regression).
- Menangani imbalance dataset dengan **SMOTE**.
- Memberikan probabilitas dan interpretasi hasil.

---

## 📁 Struktur Proyek

```
├── model/
│   ├── fraud_model.pkl           # Model Logistic Regression
│   ├── scaler.pkl                # Objek StandardScaler
│   ├── label_encoder.pkl         # Encoder untuk kolom 'type'
├── dummy_input.csv               # Contoh data dummy (20/100 transaksi)
├── predict_csv.py                # Script untuk prediksi batch (terminal)
├── predict.py                    # Script untuk prediksi personal (terminal)
├── AML.ipynb                     # Notebook pelatihan model
├── app.py                        # Aplikasi Streamlit
├── README.md                     # Dokumen ini
```

---

## ⚙️ Cara Menjalankan

### 1. Instalasi Dependensi

```bash
pip install pandas scikit-learn imbalanced-learn joblib streamlit plotly
```

### 2. Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan berjalan di:  
👉 `http://localhost:8501`

---

## 🧾 Fitur Aplikasi Streamlit

| Tab                     | Fitur                                                                 |
|------------------------|------------------------------------------------------------------------|
| **📝 Input Manual**     | Form input untuk memeriksa 1 transaksi secara manual                   |
| **📁 Upload CSV**       | Upload file CSV dan deteksi fraud secara batch                        |
| **📊 Visualisasi**      | Menampilkan pie chart hasil klasifikasi (Aman vs Mencurigakan)        |
| **📥 Download Hasil**   | Export hasil prediksi dalam bentuk CSV                                |
| **❓ Bantuan**           | Penjelasan label, format input, dan template CSV                      |

---

## 📊 Evaluasi Model

| Metric              | Train Set | Test Set |
|---------------------|-----------|----------|
| Accuracy            | 96%       | 96%      |
| ROC AUC             | 97.5%     | 97.7%    |
| Recall (Fraud)      | 86%       | 85%      |
| Precision (Fraud)   | 3%        | 3%       |

> Model memiliki generalisasi baik dan **tidak overfitting** meskipun data sangat tidak seimbang (fraud hanya <0.1%).

---

## 🔢 Penjelasan Fitur

| Fitur                  | Deskripsi                                                                 |
|------------------------|---------------------------------------------------------------------------|
| `transaction_type`     | Jenis transaksi (TRANSFER, PAYMENT, CASH_OUT, dll)                        |
| `amount`               | Jumlah nominal uang yang ditransaksikan                                   |
| `oldbalanceOrg`        | Saldo awal akun pengirim                                                  |
| `newbalanceOrig`       | Saldo akhir akun pengirim setelah transaksi                               |
| `oldbalanceDest`       | Saldo awal akun penerima                                                  |
| `newbalanceDest`       | Saldo akhir akun penerima setelah transaksi                               |
| `count_txn_by_orig`    | Total transaksi sebelumnya oleh akun pengirim                             |
| `sum_txn_by_orig`      | Total nominal dari semua transaksi pengirim                               |
| `avg_amount_by_orig`   | Rata-rata nominal transaksi oleh pengirim                                 |
| `count_unique_dest`    | Banyaknya akun penerima unik dari akun pengirim                           |
| `count_txn_by_dest`    | Jumlah transaksi yang diterima oleh akun tujuan                           |

---

## 📁 Template CSV (Wajib)

File CSV harus memiliki kolom berikut:

```
transaction_type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,count_txn_by_orig,sum_txn_by_orig,avg_amount_by_orig,count_unique_dest,count_txn_by_dest
```

✅ Contoh tersedia dalam `dummy_input.csv`

---

## 🔁 Alur Pemrosesan Data (Processing Workflow)
```
[DATA TRANSAKSI MENTAH (Kaggle)]
│
▼
[PRA-PEMROSESAN DATA]
- Feature Engineering
- Encoding Kategorikal (LabelEncoder)
- Penskalaan Fitur (StandardScaler)
│
▼
[PENANGANAN IMBALANCE DATA]
- SMOTE (pada data latih)
│
▼
[PELATIHAN MODEL]
- Split Data (Train/Test)
- Model Logistic Regression
│
▼
[EVALUASI MODEL]
- Metrik: Accuracy, ROC AUC, Recall, Precision
│
▼
[PREDIKSI TRANSAKSI BARU] <── [INPUT TRANSAKSI BARU (Manual / CSV)]
- Pra-pemrosesan (menggunakan scaler & encoder tersimpan)
│
▼
[HASIL KLASIFIKASI]
- Aman / Mencurigakan (Fraud)
- Probabilitas
```

## 🇮🇩 Catatan Lokal

Aplikasi ini dikembangkan dengan pertimbangan:
- **Bahasa Indonesia** untuk antar muka
- **Visualisasi modern** (Plotly)
- **Format edukatif** bagi mahasiswa atau pemula dalam deteksi fraud

---

## 👤 Author
**Kelompok 9**

- **Dimas Firmansyah (312210267)**
- **Aditya Putra Wijaya (312210207)**
- **Mohamad Mahdi Alethea (312210195)**

Program Studi Teknik Informatika  
Tugas Data Mining 2025

---
