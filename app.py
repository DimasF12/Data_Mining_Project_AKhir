
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model dan tools
model = joblib.load('model/fraud_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Styling
st.set_page_config(page_title="Deteksi Pencucian Uang", page_icon="💰", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💰 Deteksi Pencucian Uang (AML)</h1>", unsafe_allow_html=True)
st.markdown("### Sistem deteksi otomatis untuk transaksi mencurigakan", unsafe_allow_html=True)

# Pilihan input
tab1, tab2 = st.tabs(["📝 Input Manual", "📁 Upload CSV"])

with tab1:
    with st.form("fraud_form"):
        st.subheader("Isi Detail Transaksi di Bawah Ini:")
        
        transaction_type = st.selectbox("Jenis Transaksi", options=["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Jumlah Transaksi", min_value=0.0)
        oldbalanceOrg = st.number_input("Saldo Awal Pengirim", min_value=0.0)
        newbalanceOrig = st.number_input("Saldo Setelah Transaksi Pengirim", min_value=0.0)
        oldbalanceDest = st.number_input("Saldo Awal Penerima", min_value=0.0)
        newbalanceDest = st.number_input("Saldo Setelah Transaksi Penerima", min_value=0.0)
        count_txn_by_orig = st.number_input("Jumlah Transaksi oleh Pengirim", min_value=0)
        sum_txn_by_orig = st.number_input("Total Nominal Transaksi oleh Pengirim", min_value=0.0)
        avg_amount_by_orig = st.number_input("Rata-rata Nominal Transaksi oleh Pengirim", min_value=0.0)
        count_unique_dest = st.number_input("Jumlah Tujuan Unik oleh Pengirim", min_value=0)
        count_txn_by_dest = st.number_input("Jumlah Transaksi oleh Penerima", min_value=0)

        submitted = st.form_submit_button("🔍 Periksa Transaksi")

    if submitted:
        try:
            type_encoded = label_encoder.transform([transaction_type])[0]
            data = pd.DataFrame([{
                'type': type_encoded,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'count_txn_by_orig': count_txn_by_orig,
                'sum_txn_by_orig': sum_txn_by_orig,
                'avg_amount_by_orig': avg_amount_by_orig,
                'count_unique_dest': count_unique_dest,
                'count_txn_by_dest': count_txn_by_dest
            }])
            scaled_input = scaler.transform(data)
            pred = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0][1]

            st.markdown("### 💡 Hasil Prediksi")
            if pred == 1:
                st.error(f"⚠️ Transaksi Terdeteksi Mencurigakan! Probabilitas: {prob:.2%}")
            else:
                st.success(f"✅ Transaksi Aman. Probabilitas: {prob:.2%}")
        except Exception as e:
            st.warning(f"Gagal memproses input: {e}")
with tab2:
    st.subheader("Unggah File CSV")

    st.markdown("""
    **📌 Format Wajib Kolom:**
    - `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`,  
    - `count_txn_by_orig`, `sum_txn_by_orig`, `avg_amount_by_orig`, `count_unique_dest`, `count_txn_by_dest`

    👉 Jika belum punya, unduh template CSV di bawah:
    """)

    # Template download
    template_df = pd.DataFrame(columns=[
        'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'count_txn_by_orig',
        'sum_txn_by_orig', 'avg_amount_by_orig', 'count_unique_dest',
        'count_txn_by_dest'
    ])
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Template CSV", template_csv, file_name="template_transaksi.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df['type'] = label_encoder.transform(df['type'])
            scaled = scaler.transform(df)
            predictions = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[:, 1]

            df['Prediksi'] = predictions
            df['Probabilitas Fraud'] = probabilities

            st.markdown("### 📊 Hasil Deteksi")
            st.dataframe(df.head(20))


            # Pie Chart
            import plotly.express as px

            # Penjelasan Kolom Type
            st.markdown("#### ℹ️ Keterangan Label Type:")
            st.markdown("- `0` = Type **CASH_IN**")
            st.markdown("- `1` = Type **CASH_OUT**")
            st.markdown("- `2` = Type **DEBIT**")
            st.markdown("- `3` = Type **PAYMENT**")
            st.markdown("- `4` = Type **TRANSFER**")

            # Penjelasan label
            st.markdown("#### ℹ️ Keterangan Label Prediksi:")
            st.markdown("- `0` = Transaksi **Normal**")
            st.markdown("- `1` = Transaksi **Mencurigakan / Fraud**")

            # Hitung jumlah prediksi
            pie_data = df['Prediksi'].value_counts().reset_index()
            pie_data.columns = ['Label', 'Jumlah']
            pie_data['Label'] = pie_data['Label'].map({0: 'Aman', 1: 'Mencurigakan'})

            # Buat warna khusus per label
            warna_khusus = {'Aman': '#4CAF50', 'Mencurigakan': '#F44336'}

            fig = px.pie(
                pie_data,
                names='Label',
                values='Jumlah',
                title='Distribusi Transaksi',
                color='Label',
                color_discrete_map=warna_khusus  # Mapping label ke warna
            )
            st.plotly_chart(fig)

            csv_result = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Deteksi", csv_result, file_name="hasil_prediksi.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Terjadi kesalahan saat memproses file: {e}")
