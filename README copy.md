## Loan Approval Predictor (Streamlit)

An updated Streamlit app for predicting loan approval using a trained model (`loan_model.pkl`). This is a refreshed version of the 2024 academic project with improved UI, error handling, and documentation.

### Features
- Streamlined UI with responsive columns and sidebar info
- Cached model loading with helpful error messages
- Displays approval prediction and probability (if available)
- Clear notes on input feature encoding

### Project Structure
```
Loan Prediction Model/
  ├─ app.py                 # Streamlit application
  ├─ loan_model.pkl         # Trained model artifact (place in this folder)
  ├─ train.csv              # Training data (optional, for reference)
  └─ train_model.ipynb      # Training notebook (optional)
```

### Setup
1. Install Python 3.9+.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Ensure `loan_model.pkl` is in the project root next to `app.py`.

### Run
```bash
streamlit run app.py
```

### Inputs and Encoding
The app encodes inputs to match the 2024 training pipeline:
- Gender: Male=1, Female=0
- Married: Yes=1, No=0
- Dependents: '3+' → 3
- Education: Graduate=1, Not Graduate=0
- Self-Employed: Yes=1, No=0
- Credit History: Good=1, Bad=0
- Property Area: Urban=0, Semiurban=1, Rural=2

Numeric fields are integers: Applicant Income, Coapplicant Income, Loan Amount (in thousands), Loan Term (months).

### Model Notes
- The app attempts to show approval probability if `predict_proba` is available (e.g., LogisticRegression, RandomForest, etc.).
- If you retrain the model and change feature order/encoding, update `app.py` accordingly.

### Troubleshooting
- "Model file not found": Place `loan_model.pkl` in the same folder as `app.py`.
- "Prediction failed": Ensure the model expects 11 features in the same order as encoded in the app.

### License
For academic/educational use.


