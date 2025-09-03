# 🩺 Medical Chatbot 🤖  

This project is a **Streamlit web application** that uses **LangChain, FAISS, HuggingFace Embeddings, and Groq API** to build a medical chatbot.  
The chatbot reads a medical PDF, splits it into chunks, embeds it, and stores it in a vector database for **Retrieval-Augmented Generation (RAG)**.  

### 🌐 Live App: [Your Streamlit Link Here]([https://share.streamlit.io/](https://medicalchatbot-n7xzpc4cghrcgh57unw3fz.streamlit.app/))  

---

## 🚀 Features  

- Ask medical questions by typing OR selecting from a dropdown.  
- Retrieves the most relevant context from the medical PDF.  
- Displays detailed **token usage** (Prompt, Completion, Total, Cost).  
- Predefined sample questions for quick testing.  

---

## 🧑‍⚕️ Predefined Questions  

- What is the general prevention of Cold and flu?  
- What is First Aid and Emergency Care?  
- Tell about Future of Medicine AI and Healthcare?  

---

## 🛠 Technologies Used  

- Python 3.9+  
- Streamlit  
- LangChain  
- FAISS  
- HuggingFace Embeddings  
- Groq API  

---

## ⚙️ How It Works  

1. The medical PDF is pre-processed, split into chunks, and embedded using HuggingFace embeddings.  
2. The chunks are stored in **FAISS Vector Database** for efficient retrieval.  
3. When a question is asked, the chatbot retrieves the most relevant chunks.  
4. The retrieved context is passed to **Groq LLM** to generate the final answer.  
5. Token usage details are displayed in a collapsible dropdown.  

---

## 💬 How to Use  

1. Select a predefined question from the dropdown **OR** type your own question.  
2. The chatbot retrieves the relevant answer from the PDF.  
3. Expand the **Token Usage** dropdown to view detailed usage (Prompt tokens, Completion tokens, Total tokens, and Cost).  

---

## 📦 Requirements  

- Python 3.9+  
- Streamlit  
- LangChain  
- FAISS  
- HuggingFace Embeddings  
- Groq API  

Install all dependencies:  
```bash
pip install -r requirements.txt

## **Usage**
To run this project locally:
1. Clone the repository
2. Install the required dependencies: pip install -r requirements.txt
3. Run the Streamlit app: streamlit run app.py


## ⚠️ **Disclaimer**  

This chatbot is for **educational and informational purposes only**.  
It is **not a substitute for professional medical advice, diagnosis, or treatment**.  
Always consult a qualified healthcare professional for medical concerns.  

---

## 🔮 **Future Improvements**  

- Add support for multiple PDFs.  
- Integrate live medical knowledge APIs.  
- Improve UI with conversational history.  
- Add multilingual support.

## About the Developer
This project was developed by Santhosh Kumar M .
 You can find more about the developer here:
- GitHub: [https://github.com/Santhoshkumar099](https://github.com/Santhoshkumar099)
- LinkedIn: [https://www.linkedin.com/in/santhosh-kumar-2040ab188/](https://www.linkedin.com/in/santhosh-kumar-2040ab188/)
- Email: sksanthoshhkumar99@gmail.com

## Contributing
Contributions, issues, and feature requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)


