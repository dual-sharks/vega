import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from trading_dashboard import render_trading_dashboard
from dod_opportunities import render_dod_opportunities

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load prompts
def load_prompt(filename):
    """Load a prompt from the prompts directory"""
    prompt_path = os.path.join("prompts", filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"❌ Prompt file not found: {prompt_path}")
        return ""

# Initialize Streamlit
st.set_page_config(page_title="Vega 0.10", layout="wide")
st.title("🚀 Vega 0.10")

# Load resume from CSV instead of PDF
resume_file = "resume_v1.csv"
if not os.path.exists(resume_file):
    st.error(f"❌ Please make sure {resume_file} exists in this folder!")
    st.stop()

# Load the CSV
df = pd.read_csv(resume_file)

# Convert CSV to a string blob for embedding
resume_text = ""

# Add project-specific content
for index, row in df.iterrows():
    resume_text += f"""
Company: {row.get('Company', '')}
Role: {row.get('Role', '')}
Technologies: {row.get('Technologies', '')}
Description: {row.get('Description', '')}
Impact: {row.get('Impact', '')}

"""

# Chunk the resume text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([resume_text])

# Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Vector store
vectordb = FAISS.from_documents(
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

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["📋 Job Analysis", "📊 Trading Dashboard", "🎯 DoD Opportunities"])

with tab1:
    st.markdown("Paste an **Upwork job description** below, and I will analyze how well it matches your resume:")

    job_text = st.text_area("Job description", height=300)

    if st.button("Analyze Fit"):
        if not job_text.strip():
            st.warning("Please paste a job description before clicking Analyze.")
        else:
            with st.spinner("Analyzing..."):
                # Load the job analysis prompt
                prompt_template = load_prompt("job_analysis_prompt.txt")
                if prompt_template:
                    prompt = prompt_template.format(resume_text=resume_text, job_text=job_text)
                    answer = llm.predict(prompt)
                    st.subheader("✅ AI Analysis")
                    st.write(answer)
                else:
                    st.error("❌ Could not load job analysis prompt template.")

with tab2:
    render_trading_dashboard()

with tab3:
    render_dod_opportunities(llm)
