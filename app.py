import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit
st.set_page_config(page_title="Resume Job Match RAG", layout="wide")
st.title("🚀 Resume Job Fit Analyzer")

# Load resume from PDF
resume_file = "resume.pdf"
if not os.path.exists(resume_file):
    st.error(f"❌ Please make sure {resume_file} exists in this folder!")
    st.stop()

# PyPDFLoader is easy and reliable
loader = PyPDFLoader(resume_file)
pages = loader.load()
resume_text = "\n\n".join(page.page_content for page in pages)

# Chunk the resume text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([resume_text])

# Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Chroma store
vectordb = Chroma.from_documents(
    docs,
    embedding=embeddings
)


retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Streamlit UI
st.markdown("Paste an **Upwork job description** below, and I will analyze how well it matches your resume:")

job_text = st.text_area("Job description", height=300)

if st.button("Analyze Fit"):
    if not job_text.strip():
        st.warning("Please paste a job description before clicking Analyze.")
    else:
        with st.spinner("Analyzing..."):
            prompt = f"""
You are a resume reviewer. Given my resume and the following job posting, analyze:
- whether this is a good match for me
- what skills are missing
- what skills to highlight in my proposal
- give me a confidence rating from 0 to 100
The job posting is: {job_text}
"""
            answer = qa_chain.run(prompt)
            st.subheader("✅ AI Analysis")
            st.write(answer)
