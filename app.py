
import streamlit as st
import joblib

model = joblib.load("healthwire_week2_model_improved.joblib")

st.title("HealthWire FAQ Chatbot")
st.write("Ask a general health-related question.")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip():
        prediction = model.predict([question])[0]
        st.success(f"Predicted category: {prediction}")
    else:
        st.warning("Please enter a question.")
