import streamlit as st
import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.callbacks import get_openai_callback


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Medical_Chatbot"


# PDF Loading 
loader = PyPDFLoader("Medical_document_chatbot.pdf")
docs = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

# Create embeddings and FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstoredb = FAISS.from_documents(final_documents, embeddings)
retriever = vectorstoredb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# LLM Setup
llm = ChatGroq(temperature=0.3, model_name="openai/gpt-oss-20b")

system_prompt = """
Role:
You are an AI medical assistant. Your job is to answer questions strictly using the provided medical knowledge base (PDF data).

INSTRUCTIONS:
- Always list **all prevention measures** or relevant details available in the context, not just one.
- If multiple chunks contain answers, merge them into a clear, structured response (use bullet points where helpful).
- If the answer is not found in the PDF, respond politely with:
    "I am sorry, I do not have that information in my records."
- Keep responses concise but **comprehensive**.
- Strictly do not provide information outside the PDF.

Context from PDF:
{context}
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ('human', "{input}")
])

# Create chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Knowledge Chatbot")

sample_questions = [
    "What is the general prevention of Cold and flue ?",
    "What is First Aid and Emergency Care ?",
    "Tell about Future of Medicine AI and Healthcare?"
]

selected_question = st.selectbox("💡 Choose a sample question (optional):", [""] + sample_questions)

# Input box for user question
user_question = st.text_input("Or type your own question here:")

# Decide which question to use
final_question = user_question if user_question.strip() else selected_question

if st.button("Get Answer"):
    if final_question:
        with get_openai_callback() as cb:
            response = rag_chain.invoke({"input": final_question})

            st.subheader("✅ Answer")
            st.write(response["answer"])

            # Token usage expandable section
            with st.expander("📊 Token Usage Details"):
                st.write(f"**Prompt Tokens:** {cb.prompt_tokens}")
                st.write(f"**Completion Tokens:** {cb.completion_tokens}")
                st.write(f"**Total Tokens:** {cb.total_tokens}")
                st.write(f"**Estimated Cost:** ${cb.total_cost:.5f}")
    else:
        st.warning("Please select or type a question to proceed.")


