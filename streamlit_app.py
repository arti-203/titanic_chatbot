import streamlit as st
import requests

st.set_page_config(page_title="Titanic Chatbot", page_icon="🚢")

st.title("🚢 Titanic Data Chatbot")
st.write("Ask me about the passengers! (e.g., 'What percentage were male?', 'Show me a histogram of ages')")

# Input box
user_input = st.text_input("Ask a question:", key="input")

if st.button("Ask"):
    if user_input:
        with st.spinner("Analyzing data..."):
            try:
                # Call the FastAPI backend
                response = requests.post("http://localhost:8000/chat", json={"question": user_input})
                data = response.json()
                
                if data["type"] == "text":
                    st.success(data["content"])
                elif data["type"] == "image":
                    # Display the base64 image
                    st.image(f"data:image/png;base64,{data['content']}", caption="Visual Insight")
                else:
                    st.warning("Unexpected response format")
                    
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
                st.info("Make sure the FastAPI backend (main.py) is running on port 8000.")