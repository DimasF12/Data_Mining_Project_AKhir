
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
├── model
    ├── fraud_model.pkl           # Model Logistic Regression
    ├── scaler.pkl                # Objek StandardScaler
    ├── label_encoder.pkl         # Encoder untuk kolom 'type'
├── dummy_input.csv           # Contoh data dummy (20 transaksi)
├── predict_csv.py            # Script untuk prediksi batch
├── predict.py                # Script untuk prediksi personal
├── AML.ipynb                 # file note book untuk pelatihan model
├── README.md                 # Dokumen ini
```

---

## ⚙️ Cara Menjalankan

### 1. Persiapan
Install package yang dibutuhkan:
```bash
pip install pandas scikit-learn imbalanced-learn joblib
```

### 2. Menjalankan Prediksi Manual
Jalankan script `predict.py`, pastikan file model (`*.pkl`) ada di direktori yang sama:
```bash
python predict.py
```

Contoh pemanggilan fungsi:
```python
predict_fraud(
    transaction_type='TRANSFER',
    amount=700000.0,
    oldbalanceOrg=900000.0,
    newbalanceOrig=0.0,
    oldbalanceDest=0.0,
    newbalanceDest=900000.0,
    count_txn_by_orig=20,
    sum_txn_by_orig=5000000.0,
    avg_amount_by_orig=250000.0,
    count_unique_dest=10,
    count_txn_by_dest=3
)
```

Atau jalankan `predict_batch_from_csv('dummy_input.csv')` untuk memproses banyak data sekaligus.

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

## 📁 Contoh Dummy Data (`dummy_input.csv`)

```csv
transaction_type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,count_txn_by_orig,sum_txn_by_orig,avg_amount_by_orig,count_unique_dest,count_txn_by_dest
TRANSFER,700000,900000,0,0,900000,20,5000000,250000,10,3
PAYMENT,12000,13000,1000,0,0,10,120000,12000,5,0
...
```

---

## 🇮🇩 Catatan Lokal

Model ini didesain untuk **penggunaan lokal dan edukasi**, dengan konteks pengguna di Indonesia. Untuk keperluan produksi/nyata:
- Precision harus ditingkatkan untuk mengurangi false positive.
- Model bisa diganti atau ditingkatkan menggunakan XGBoost atau Random Forest.
- Sistem bisa diintegrasikan ke API/Frontend untuk otomasi.

---

## 👤 Author

**Dimas Firmansyh**
**Aditya Putra Wijaya**
**Mohamad Mahdi Alethea**
Program Studi Teknik Informatika  
Tugas Data Mining 2025

---
