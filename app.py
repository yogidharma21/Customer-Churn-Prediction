import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df.dropna(subset=["TotalCharges"]).copy()


model = load_model()
df = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Churn Analytics")

page = st.sidebar.radio(
    "Menu",
    [
        "Dashboard Overview",
        "Churn Prediction"
    ]
)

# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.title("📊 Customer Churn Dashboard")

    st.caption(
        "Telco Customer Churn — Analytics & Predictive Insights"
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_customers = len(df)

    churn_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churn_customers /
        total_customers
    ) * 100

    avg_monthly = df["MonthlyCharges"].mean()

    avg_tenure = df["tenure"].mean()

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
        f"{avg_monthly:.2f}"
    )

    col4.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} months"
    )

    st.divider()

    # --------------------------------------------------------
    # CHURN OVERVIEW
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        churn_data = (
            df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_data.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_data,
            names="Churn",
            values="Customers",
            title="Customer Churn Distribution",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        contract_churn = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            contract_churn,
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

    st.subheader("Churn Rate by Contract")

    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
        .sort_values("Churn Rate", ascending=False)
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

    st.subheader("Churn Rate by Internet Service")

    internet_rate = (
        df.groupby("InternetService")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
        .sort_values("Churn Rate", ascending=False)
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
        opacity=0.6,
        title="Tenure vs Monthly Charges by Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# CHURN PREDICTION
# ============================================================

else:

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Masukkan karakteristik pelanggan untuk melihat "
        "probabilitas churn dan rekomendasi retensi."
    )

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=70.00
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=9000.0,
            value=1000.0
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    ):

        customer = pd.DataFrame([{
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
        }])

        probability = model.predict_proba(
            customer
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

            risk = "Low Risk"

        elif probability < 0.50:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

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
            risk
        )

        col3.metric(
            "Prediction",
            "Churn" if prediction == "Yes"
            else "No Churn"
        )

        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.progress(
            float(probability)
        )

        # ----------------------------------------------------
        # RECOMMENDATION
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
                "Evaluasi paket dan biaya bulanan pelanggan."
            )

        if internet_service == "Fiber optic":
            recommendations.append(
                "Periksa kepuasan dan pengalaman pengguna "
                "Fiber optic."
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
                "Lakukan monitoring rutin dan gunakan "
                "probabilitas churn sebagai indikator risiko."
            )

        for recommendation in recommendations:
            st.info(recommendation)

        # ----------------------------------------------------
        # BUSINESS NOTE
        # ----------------------------------------------------

        st.caption(
            f"Prediction menggunakan threshold {threshold:.2f}. "
            "Threshold ini dapat disesuaikan dengan biaya "
            "dan strategi retensi perusahaan."
        )
