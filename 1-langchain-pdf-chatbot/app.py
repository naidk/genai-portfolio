import os
from dotenv import load_dotenv
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import tempfile

# 🔐 Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# 📄 Streamlit setup
st.set_page_config(page_title="📄 PDF ChatBot", layout="wide")
st.title("🧠 LangChain PDF ChatBot with Memory")

# 📁 File uploader
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    # ✅ Use a temporary file instead of saving to data/
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        temp_pdf_path = tmp_file.name

    # 📄 Load PDF directly from the temporary file
    loader = PyPDFLoader(temp_pdf_path)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(pages, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your PDF:")
    if query:
        result = qa_chain.run(query)
        st.session_state.chat_history.append(("🧑", query))
        st.session_state.chat_history.append(("🤖", result))

    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")
