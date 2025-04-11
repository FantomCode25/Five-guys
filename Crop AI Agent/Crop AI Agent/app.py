import streamlit as st  
from rag_agent import qa  

st.title("🌱 CropMind (Local AI)")  
soil_ph = st.number_input("Enter soil pH:", min_value=0.0, max_value=14.0, value=6.5)  
question = st.text_input("Ask a farming question...")  

if question:  
    try:  
        answer = qa.run(question)  
        st.write("📚 **Answer**:", answer)  

        # Add pH-based warnings  
        if soil_ph < 5.5:  
            st.warning("⚠️ Add lime to raise pH!")  
    except Exception as e:  
        st.error("AI is taking a nap 😴 Try again or simplify your question.")  