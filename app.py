import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("loan_model.pkl")
    except FileNotFoundError:
        st.error("Model file `loan_model.pkl` not found. Place it in the app directory.")
        return None
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None

model = load_model()

# Minimal styling tweaks for buttons and headers
st.markdown(
    """
    <style>
    h1 { text-align: center; margin-bottom: 0.25rem; }
    .stButton button { background-color: #4CAF50; color: white; border: 0; border-radius: 8px; }
    .stButton button:hover { background-color: #45A049; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar info
with st.sidebar:
    st.subheader("About")
    st.write("Predict loan approval using a model trained on historical data.")
    st.caption("Note: Inputs and encoding must match training features from 2024 project.")

# Title and intro
st.title("🏦 Loan Approval Predictor")
st.write("Fill the form and click Predict to estimate approval.")

# Inputs using responsive columns
st.write("### Applicant Details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender", help="Select the applicant's gender.")
    married = st.selectbox("Married", ["Yes", "No"], key="married", help="Is the applicant married?")
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="dependents", help="Number of dependents.")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="education", help="Highest education.")
    self_employed = st.selectbox("Self-Employed", ["Yes", "No"], key="self_employed", help="Is the applicant self-employed?")
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, step=1, key="income", help="Monthly income (₹).", format="%d")
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=1, key="co_income", help="Monthly co-applicant income (₹).", format="%d")
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, step=1, key="loan_amt", help="Requested amount in thousands.", format="%d")
    loan_term = st.number_input("Loan Term (in months)", min_value=0, step=1, key="loan_term", help="Duration in months.", format="%d")
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"], key="credit_hist", help="1=Good, 0=Bad")

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], key="property", help="Property location category.")

# Prepare features – keep original encoding to match the trained model
try:
    input_data = np.array([
        [
            1 if gender == "Male" else 0,
            1 if married == "Yes" else 0,
            int(dependents.replace("+", "")),
            1 if education == "Graduate" else 0,
            1 if self_employed == "Yes" else 0,
            int(applicant_income),
            int(coapplicant_income),
            int(loan_amount),
            int(loan_term),
            1 if credit_history == "Good (1)" else 0,
            0 if property_area == "Urban" else (1 if property_area == "Semiurban" else 2),
        ]
    ])
except Exception as exc:
    st.error(f"Invalid input: {exc}")
    input_data = None

predict_clicked = st.button("🔮 Predict", use_container_width=True, type="primary")

if predict_clicked:
    if model is None:
        st.stop()
    if input_data is None:
        st.stop()

    try:
        prediction = model.predict(input_data)
        st.write("### Prediction Result:")
        if prediction[0] == 1:
            st.success("🎉 Loan Approved ✅")
        else:
            st.error("🚫 Loan Not Approved ❌")

        # If classifier exposes probabilities, show them
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            # Assuming class 1 is approval
            approval_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            st.progress(min(max(approval_prob, 0.0), 1.0))
            st.caption(f"Estimated approval probability: {approval_prob:.1%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Notes on feature encoding"):
    st.markdown(
        "- Gender: Male=1, Female=0\n"
        "- Married: Yes=1, No=0\n"
        "- Dependents: '3+' → 3\n"
        "- Education: Graduate=1, Not Graduate=0\n"
        "- Self-Employed: Yes=1, No=0\n"
        "- Credit History: Good=1, Bad=0\n"
        "- Property Area: Urban=0, Semiurban=1, Rural=2"
    )
