import joblib
import pandas as pd
import numpy as np

def prediksi_fraud_manual():
    # Muat model dan preprocessor
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Input manual (bisa ganti dari input() kalau mau interaktif)
    jenis_transaksi = 'TRANSFER'
    jumlah = 700000.0
    saldo_awal_pengirim = 900000.0
    saldo_akhir_pengirim = 0.0
    saldo_awal_penerima = 0.0
    saldo_akhir_penerima = 900000.0
    jumlah_transaksi_pengirim = 20
    total_uang_dikirim_pengirim = 5000000.0
    rata_rata_pengiriman_pengirim = 250000.0
    jumlah_tujuan_unik = 10
    jumlah_transaksi_penerima = 3

    # Encode kategori
    jenis_encoded = label_encoder.transform([jenis_transaksi])[0]

    # Buat DataFrame
    data_input = pd.DataFrame([{
        'type': jenis_encoded,
        'amount': jumlah,
        'oldbalanceOrg': saldo_awal_pengirim,
        'newbalanceOrig': saldo_akhir_pengirim,
        'oldbalanceDest': saldo_awal_penerima,
        'newbalanceDest': saldo_akhir_penerima,
        'count_txn_by_orig': jumlah_transaksi_pengirim,
        'sum_txn_by_orig': total_uang_dikirim_pengirim,
        'avg_amount_by_orig': rata_rata_pengiriman_pengirim,
        'count_unique_dest': jumlah_tujuan_unik,
        'count_txn_by_dest': jumlah_transaksi_penerima
    }])

    # Scaling
    data_scaled = scaler.transform(data_input)

    # Prediksi
    hasil = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    print("=== Hasil Prediksi ===")
    print(f"Tipe Transaksi  : {jenis_transaksi}")
    print(f"Prediksi        : {'FRAUD 🛑' if hasil == 1 else 'Normal ✅'}")
    print(f"Probabilitas    : {prob*100:.2f}%")

if __name__ == "__main__":
    prediksi_fraud_manual()
