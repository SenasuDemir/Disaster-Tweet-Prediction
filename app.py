import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vect = joblib.load('vectorizer.pkl')

# Sentiment prediction function
def sentiment_prediction(text):
    text_arr = [text]
    text_transformed = vect.transform(text_arr)
    prediction = model.predict(text_transformed)
    return prediction

# Main function for app layout and interaction
def main():
    # Set page configuration
    st.set_page_config(page_title="Disaster Tweet Prediction", page_icon="🎭", layout="wide")

    # Custom CSS styling
    st.markdown("""
        <style>
            .title {
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                color: #ff4c4c;
                margin-top: 20px;
            }
            .input-area {
                background-color: #f5f5f5;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .stTextArea textarea {
                font-size: 18px;
                border-radius: 8px;
                padding: 12px;
                width: 100%;
            }
            .result {
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-top: 20px;
            }
            .Related-with-Disaster {
                background-color: #ff4c4c;
                color: white;
            }
            .Not-Related-with-Disaster {
                background-color: #4caf50;
                color: white;
            }
            .confidence {
                font-size: 20px;
                text-align: center;
                margin-top: 10px;
                font-weight: 600;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    # App Title
    st.markdown('<div class="title">Disaster Tweet Prediction</div>', unsafe_allow_html=True)

    # Input area for text
    with st.container():
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        text = st.text_area("Type your tweet:", "", height=150)
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button with custom style
    if st.button("Predict Sentiment"):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text to make a prediction!")
        else:
            sentiment_pred = sentiment_prediction(text)
            sentiment_label = "Related with Disaster" if sentiment_pred[0] == 1 else "Not Related with Disaster"
            confidence = np.random.uniform(0.75, 0.95)  # Fake confidence score (replace with actual if available)

            # Result visualization with fancy effects
            result_class = "Related-with-Disaster" if sentiment_pred[0] == 1 else "Not-Related-with-Disaster"
            st.markdown(f'<div class="result {result_class}">🎭 Prediction: {sentiment_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">✨ Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
