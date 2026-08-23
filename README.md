# Telco Customer Churn Prediction & Retention Analytics

End-to-end data analytics & machine learning project untuk memprediksi customer churn pada perusahaan telekomunikasi, sekaligus menerjemahkan hasil model menjadi rekomendasi bisnis yang actionable bagi tim retensi.

**Notebook:** [`Customer_Churn_Prediction_Analysis.ipynb`](./Customer_Churn_Prediction_Analysis.ipynb)

---

## Ringkasan Project

Perusahaan telekomunikasi kehilangan pendapatan setiap kali pelanggan berhenti berlangganan (churn), namun tidak semua pelanggan memiliki risiko yang sama. Project ini membangun model klasifikasi untuk memprediksi kemungkinan seorang pelanggan akan churn, sekaligus mengidentifikasi faktor-faktor yang paling berasosiasi dengan keputusan tersebut — sehingga tim retensi dapat memprioritaskan pelanggan berisiko tinggi alih-alih menyasar seluruh basis pelanggan secara merata.

## 🚀 Live Dashboard

Project ini juga tersedia dalam bentuk interactive dashboard menggunakan **Streamlit**.

🔗 **[Open Customer Churn Dashboard →](https://customer-churn-prediction-44bkaa6oyfwyflshxiajsk.streamlit.app/)**

Dashboard menyediakan:

- 📊 **Overview** — customer churn rate, contract, internet service, tenure, dan monthly charges.
- 🔮 **Churn Prediction** — memasukkan karakteristik pelanggan untuk mendapatkan probabilitas churn dan risk level.
- 💡 **Business Insights** — key findings, model performance, threshold analysis, dan rekomendasi retensi.

> **Note:** Dashboard ini merupakan prototype predictive analytics. Hasil prediksi digunakan sebagai alat bantu pengambilan keputusan, bukan keputusan otomatis.

## Dataset

- **Sumber:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, via `kagglehub`)
- **Ukuran:** 7.043 baris awal → **7.032 pelanggan** setelah data cleaning
- **Target:** `Churn` (Yes/No) — binary classification
- **Fitur:** informasi demografis (gender, senior citizen, dependents), layanan yang digunakan (internet, phone, streaming, tech support, dll), jenis kontrak, metode pembayaran, tenure, dan tagihan bulanan

## Alur Analisis

1. **Domain Proyek & Business Understanding** — konteks bisnis churn, problem statement, dan tujuan project
2. **Data Understanding** — struktur dataset, data dictionary, data quality check
3. **Data Cleaning** — penanganan 11 baris `TotalCharges` kosong (tersimpan sebagai spasi), pengecekan duplikasi
4. **Exploratory Data Analysis** — univariate, bivariate, dan multivariate analysis, tiap visualisasi disertai insight & interpretasi bisnis
5. **Data Preparation** — feature selection (termasuk menangani multikolinearitas pada `TotalCharges`), encoding, scaling, train-test split (stratified, 80:20)
6. **Modeling** — perbandingan 4 algoritma: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
7. **Evaluation** — confusion matrix, ROC curve, precision-recall curve, threshold analysis
8. **Model Interpretation** — feature importance berbasis koefisien Logistic Regression
9. **Business Insight & Recommendation** — key findings, customer risk profile, rekomendasi retensi, churn risk scoring

## Hasil Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0,7939 | 0,6346 | 0,5294 | 0,5773 | **0,8402** |
| **Logistic Regression** | **0,8031** | **0,6456** | **0,5749** | **0,6082** | 0,8336 |
| Random Forest | 0,7783 | 0,6020 | 0,4893 | 0,5398 | 0,8117 |
| Decision Tree | 0,7200 | 0,4760 | — | — | — |

**Logistic Regression dipilih sebagai model utama** meski ROC-AUC-nya sedikit di bawah Gradient Boosting — karena unggul di Accuracy, Precision, Recall, dan F1-Score, sekaligus lebih mudah diinterpretasikan lewat koefisiennya untuk kebutuhan business insight.

## Faktor Paling Berpengaruh terhadap Churn

Berdasarkan koefisien Logistic Regression:

**Meningkatkan risiko churn:**
- `InternetService: Fiber optic` (koefisien tertinggi, 1,15)
- Layanan add-on seperti `StreamingTV`, `MultipleLines`, `StreamingMovies`
- Metode pembayaran `Electronic check`

**Menurunkan risiko churn:**
- `Contract: Two year` / `One year` (kontrak jangka panjang)
- `tenure` yang lebih panjang
- Berlangganan `TechSupport` dan `OnlineSecurity`

*Catatan: seluruh hubungan di atas bersifat asosiatif (association), bukan hubungan sebab-akibat (causal).*

## Business Recommendations

1. Fokus onboarding & follow-up aktif pada pelanggan baru (tenure pendek)
2. Dorong migrasi pelanggan Month-to-month ke kontrak jangka panjang lewat insentif
3. Evaluasi paket/harga untuk pelanggan dengan `MonthlyCharges` tinggi
4. Investigasi kepuasan pelanggan Fiber optic (harga, kualitas layanan, ekspektasi)
5. Gunakan skor probabilitas churn dari model untuk memprioritaskan outreach tim retensi

Detail lengkap key insights, customer risk profile, churn risk scoring, dan business priority matrix ada di section 9 pada notebook.

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `kagglehub`

## Keterbatasan

- Dataset bersifat historis (snapshot), pola pelanggan dapat berubah seiring waktu
- Seluruh temuan bersifat asosiatif, bukan hubungan kausal
- Threshold klasifikasi belum divalidasi terhadap biaya bisnis riil (biaya outreach vs biaya kehilangan pelanggan)
- Model perlu divalidasi dengan data baru sebelum digunakan secara operasional

## Cara Menjalankan

1. Clone repo ini
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn kagglehub`
3. Jalankan `Customer_Churn_Prediction_Analysis.ipynb` dari cell paling atas (notebook otomatis mengunduh dataset lewat `kagglehub`)

---

**Author:** [Yogi Dharma Susanto](https://github.com/yogidharma21) — [Portfolio](https://yogi-dharma-portfolio.vercel.app/)
