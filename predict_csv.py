import pandas as pd
import joblib

def prediksi_dari_csv(file_csv):
    # Load model & preprocessing
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Baca CSV
    df = pd.read_csv(file_csv)

    # Encode kolom 'type'
    df['type'] = label_encoder.transform(df['type'])

    # Scaling
    data_scaled = scaler.transform(df)

    # Prediksi
    prediksi = model.predict(data_scaled)
    probabilitas = model.predict_proba(data_scaled)[:, 1]

    # Tambahkan hasil ke dataframe
    df['prediksi'] = ['FRAUD 🛑' if p == 1 else 'Normal ✅' for p in prediksi]
    df['probabilitas_fraud'] = [f"{prob*100:.2f}%" for prob in probabilitas]

    # Tampilkan hasil
    print("\n=== HASIL PREDIKSI DARI FILE ===")
    print(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'prediksi', 'probabilitas_fraud']])

    # (Opsional) Simpan ke file baru
    df.to_csv('hasil_prediksi.csv', index=False)
    print("\n✅ Hasil disimpan ke 'hasil_prediksi.csv'")

if __name__ == "__main__":
    prediksi_dari_csv("dummy.csv")
