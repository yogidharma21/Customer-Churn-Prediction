import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .risk-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            "Dataset tidak ditemukan.\n\n"
            "Pastikan file berada di:\n"
            "`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`"
        )

        st.stop()

    data = pd.read_csv(DATA_PATH)

    # Convert TotalCharges
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # Remove invalid TotalCharges
    data = data.dropna(
        subset=["TotalCharges"]
    ).copy()

    return data


df = load_data()


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


MODEL_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[MODEL_FEATURES]
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    # Test evaluation
    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_test_binary = (
        y_test == "Yes"
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label="Yes"
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label="Yes"
        ),

        "F1-Score": f1_score(
            y_test,
            y_pred,
            pos_label="Yes"
        ),

        "ROC-AUC": roc_auc_score(
            y_test_binary,
            y_prob
        )
    }

    return (
        pipeline,
        metrics
    )


model, metrics = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Churn Analytics")

st.sidebar.caption(
    "Telco Customer Churn"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🔮 Churn Prediction",
        "💡 Business Insights"
    ]
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.markdown(
        '<div class="main-title">'
        '📊 Customer Churn Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Telco customer analytics and churn prediction'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_customers = len(df)

    churn_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churn_customers
        / total_customers
        * 100
    )

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    avg_tenure = (
        df["tenure"].mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


    col2.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


    col3.metric(
        "Avg Monthly Charges",
        f"{avg_monthly_charges:.2f}"
    )


    col4.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} months"
    )


    st.divider()


    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        churn_distribution = (
            df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_distribution.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_distribution,
            names="Churn",
            values="Customers",
            hole=0.45,
            title="Customer Churn Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        contract_distribution = (
            df.groupby(
                ["Contract", "Churn"]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        fig = px.bar(
            contract_distribution,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Customers by Contract & Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHURN RATE BY CONTRACT
    # --------------------------------------------------------

    st.subheader(
        "Churn Rate by Contract"
    )

    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Churn Rate"
        )
        .sort_values(
            "Churn Rate",
            ascending=False
        )
    )

    fig = px.bar(
        contract_rate,
        x="Contract",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Contract"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # INTERNET SERVICE
    # --------------------------------------------------------

    st.subheader(
        "Churn Rate by Internet Service"
    )

    internet_rate = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Churn Rate"
        )
        .sort_values(
            "Churn Rate",
            ascending=False
        )
    )

    fig = px.bar(
        internet_rate,
        x="InternetService",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Internet Service"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TENURE VS MONTHLY CHARGES
    # --------------------------------------------------------

    st.subheader(
        "Tenure vs Monthly Charges"
    )

    fig = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        opacity=0.55,
        title="Tenure vs Monthly Charges by Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.markdown(
        '<div class="main-title">'
        '🔮 Customer Churn Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter customer information to estimate churn risk.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Customer Profile"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            sorted(
                df["gender"]
                .dropna()
                .unique()
                .tolist()
            )
        )


        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No"
        )


        partner = st.selectbox(
            "Partner",
            sorted(
                df["Partner"]
                .unique()
                .tolist()
            )
        )


        dependents = st.selectbox(
            "Dependents",
            sorted(
                df["Dependents"]
                .unique()
                .tolist()
            )
        )


        tenure = st.slider(
            "Tenure (months)",
            min_value=int(
                df["tenure"].min()
            ),
            max_value=int(
                df["tenure"].max()
            ),
            value=12
        )


    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            sorted(
                df["PhoneService"]
                .unique()
                .tolist()
            )
        )


        multiple_lines = st.selectbox(
            "Multiple Lines",
            sorted(
                df["MultipleLines"]
                .unique()
                .tolist()
            )
        )


        internet_service = st.selectbox(
            "Internet Service",
            sorted(
                df["InternetService"]
                .unique()
                .tolist()
            )
        )


        online_security = st.selectbox(
            "Online Security",
            sorted(
                df["OnlineSecurity"]
                .unique()
                .tolist()
            )
        )


        online_backup = st.selectbox(
            "Online Backup",
            sorted(
                df["OnlineBackup"]
                .unique()
                .tolist()
            )
        )


    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            sorted(
                df["DeviceProtection"]
                .unique()
                .tolist()
            )
        )


        tech_support = st.selectbox(
            "Tech Support",
            sorted(
                df["TechSupport"]
                .unique()
                .tolist()
            )
        )


        streaming_tv = st.selectbox(
            "Streaming TV",
            sorted(
                df["StreamingTV"]
                .unique()
                .tolist()
            )
        )


        streaming_movies = st.selectbox(
            "Streaming Movies",
            sorted(
                df["StreamingMovies"]
                .unique()
                .tolist()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(
            "Contract",
            sorted(
                df["Contract"]
                .unique()
                .tolist()
            )
        )


    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            sorted(
                df["PaperlessBilling"]
                .unique()
                .tolist()
            )
        )


    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            sorted(
                df["PaymentMethod"]
                .unique()
                .tolist()
            )
        )


    col1, col2 = st.columns(2)


    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0,
            step=50.0
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        customer_data = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = model.predict_proba(
            customer_data
        )[0, 1]


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        threshold = 0.40


        prediction = (
            "Yes"
            if probability >= threshold
            else "No"
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability < 0.30:

            risk_level = "Low Risk"

        elif probability < 0.50:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )


        col2.metric(
            "Risk Level",
            risk_level
        )


        col3.metric(
            "Prediction",
            "Churn"
            if prediction == "Yes"
            else "No Churn"
        )


        st.progress(
            min(
                max(
                    float(probability),
                    0.0
                ),
                1.0
            )
        )


        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk_level == "High Risk":

            st.error(
                "⚠️ Pelanggan memiliki risiko churn "
                "yang relatif tinggi."
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "⚠️ Pelanggan memiliki risiko churn "
                "tingkat menengah."
            )

        else:

            st.success(
                "✅ Pelanggan memiliki risiko churn "
                "yang relatif rendah."
            )


        # ----------------------------------------------------
        # RETENTION RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Retention Recommendation"
        )


        recommendations = []


        if tenure < 12:

            recommendations.append(
                "Prioritaskan onboarding dan follow-up "
                "karena pelanggan masih berada pada "
                "fase awal berlangganan."
            )


        if contract == "Month-to-month":

            recommendations.append(
                "Pertimbangkan penawaran untuk "
                "berpindah ke kontrak jangka panjang."
            )


        if monthly_charges >= 74:

            recommendations.append(
                "Evaluasi paket dan biaya bulanan "
                "pelanggan."
            )


        if internet_service == "Fiber optic":

            recommendations.append(
                "Periksa kepuasan dan pengalaman "
                "pengguna Fiber optic."
            )


        if tech_support == "No":

            recommendations.append(
                "Pertimbangkan menawarkan Tech Support "
                "sebagai bagian dari strategi retensi."
            )


        if online_security == "No":

            recommendations.append(
                "Pertimbangkan menawarkan Online Security "
                "kepada pelanggan berisiko tinggi."
            )


        if not recommendations:

            recommendations.append(
                "Lakukan monitoring rutin berdasarkan "
                "probabilitas churn pelanggan."
            )


        for recommendation in recommendations:

            st.info(
                recommendation
            )


        st.caption(
            "Model menggunakan threshold 0.40. "
            "Threshold tersebut dapat disesuaikan "
            "dengan kebutuhan bisnis."
        )


# ============================================================
# PAGE 3 — BUSINESS INSIGHTS
# ============================================================

else:

    st.markdown(
        '<div class="main-title">'
        '💡 Business Insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Key findings and retention recommendations'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "Key Findings"
    )


    st.markdown(
        """
        ### 1. Tenure Pendek Lebih Berisiko

        Pelanggan yang churn memiliki tenure yang lebih pendek
        dibandingkan pelanggan yang tetap bertahan.

        ### 2. Kontrak Month-to-month Perlu Diperhatikan

        Pelanggan dengan kontrak Month-to-month menunjukkan
        pola churn yang lebih tinggi dibandingkan kontrak
        jangka panjang.

        ### 3. Monthly Charges Lebih Tinggi pada Pelanggan Churn

        Pelanggan yang churn memiliki rata-rata MonthlyCharges
        lebih tinggi dibandingkan pelanggan yang tidak churn.

        ### 4. Fiber Optic Perlu Dianalisis Lebih Lanjut

        Penggunaan Fiber optic memiliki kontribusi positif
        yang cukup besar dalam model Logistic Regression.

        ### 5. Layanan Tambahan Menunjukkan Pola Retensi yang Lebih Baik

        TechSupport dan OnlineSecurity memiliki koefisien
        negatif dalam model.
        """
    )


    # --------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader(
        "Business Recommendations"
    )


    recommendations_df = pd.DataFrame({
        "Priority": [
            "High",
            "High",
            "Medium",
            "Medium"
        ],

        "Recommendation": [
            "Fokus pada onboarding pelanggan baru",
            "Dorong migrasi dari Month-to-month",
            "Evaluasi pengalaman pelanggan Fiber optic",
            "Pertimbangkan bundling TechSupport & OnlineSecurity"
        ],

        "Target": [
            "Pelanggan dengan tenure pendek",
            "Pelanggan Month-to-month",
            "Pengguna Fiber optic",
            "Pelanggan berisiko tinggi"
        ]
    })


    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Baseline Model Performance"
    )


    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": list(metrics.values())
    })


    metrics_df["Score"] = metrics_df[
        "Score"
    ].round(4)


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    st.subheader(
        "Recommended Threshold"
    )


    threshold_data = pd.DataFrame({
        "Threshold": [
            0.30,
            0.40,
            0.50
        ],

        "Precision": [
            0.5127,
            0.5779,
            0.6505
        ],

        "Recall": [
            0.7567,
            0.6845,
            0.5722
        ],

        "F1-Score": [
            0.6112,
            0.6267,
            0.6088
        ]
    })


    fig = px.line(
        threshold_data,
        x="Threshold",
        y=[
            "Precision",
            "Recall",
            "F1-Score"
        ],
        markers=True,
        title="Threshold Analysis"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        "Threshold 0.40 memberikan F1-Score tertinggi "
        "pada pengujian baseline, sementara threshold "
        "0.30 memberikan Recall tertinggi."
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.caption(
        "Dashboard ini merupakan prototype predictive analytics. "
        "Prediksi model sebaiknya digunakan sebagai alat bantu "
        "pengambilan keputusan dan bukan sebagai keputusan otomatis."
    )
