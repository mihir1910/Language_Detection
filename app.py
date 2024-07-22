import joblib
import streamlit as st

st.title("Language Detection App")

st.write("""
This app predicts the language of a given text.
""")

text_input = st.text_area("Enter text:", "")

if st.button("Predict"):
    if text_input:
        # Load the model and vectorizer
        model = joblib.load('language_detection_model.pkl')
        cv = joblib.load('count_vectorizer.pkl')

        input_data = cv.transform([text_input])

        # Predict the language
        prediction = model.predict(input_data)[0]

        st.write(f"Predicted Language: {prediction}")

