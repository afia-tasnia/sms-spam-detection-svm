"""Streamlit demo: type an SMS and see whether the model thinks it is spam.

Run:  streamlit run app.py
(first run the notebook once so that models/sms_spam_svm_model.pkl exists, or use the one in the repo)
"""
import joblib
import streamlit as st

MODEL_PATH = "models/sms_spam_svm_model.pkl"

st.set_page_config(page_title="SMS Spam Detector", page_icon="📱")
st.title("📱 SMS Spam Detector")
st.caption("Linear SVM on character + word TF-IDF features (SMS Spam Collection dataset)")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}. Run SMS_Spam_SVM_final.ipynb first to create it.")
    st.stop()

text = st.text_area("Type or paste an SMS message:", height=120,
                    placeholder="e.g. URGENT! You have won a 1000 prize. Call 09061701461 now")

if st.button("Check message", type="primary"):
    if not text.strip():
        st.warning("Please enter a message first.")
    else:
        score = float(model.decision_function([text])[0])
        if score > 0:
            st.error(f"🚨 SPAM  (score {score:+.2f})")
        else:
            st.success(f"✅ HAM / normal message  (score {score:+.2f})")
        st.caption("Score is the SVM's distance from the decision boundary: above 0 = spam; "
                   "the further from 0, the more confident.")
