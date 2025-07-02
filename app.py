import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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

# Streamlit UI
st.markdown("Paste an **Upwork job description** below, and I will analyze how well it matches your resume:")

job_text = st.text_area("Job description", height=300)

if st.button("Analyze Fit"):
    if not job_text.strip():
        st.warning("Please paste a job description before clicking Analyze.")
    else:
        with st.spinner("Analyzing..."):
            prompt = f"""
You are an expert resume and job matching analyst with deep experience in evaluating candidate-job fit. Your task is to provide a comprehensive analysis of how well the candidate's resume matches the job requirements.

JOB POSTING TO ANALYZE:
{job_text}

ANALYSIS INSTRUCTIONS:

1. **SKILL MATCH ANALYSIS** (Be very specific):
   - Extract key required skills/qualifications from the job posting
   - Compare each requirement against the resume content
   - For each skill: state if it's present, partially present, or missing
   - Be strict - only count skills that are clearly demonstrated in the resume

2. **EXPERIENCE RELEVANCE**:
   - Evaluate if the candidate's work experience directly relates to the job requirements
   - Consider industry relevance, role similarity, and project scope
   - Distinguish between "related" and "directly relevant" experience

3. **EDUCATION & CERTIFICATIONS**:
   - Check if required degrees/certifications are present
   - Evaluate if the field of study is relevant to the job requirements

4. **CONFIDENCE RATING CRITERIA** (0-100 scale):
   - 0-20: No relevant skills or experience, completely different field
   - 21-40: Some tangential skills but major gaps in core requirements
   - 41-60: Partial match with some relevant skills but missing key requirements
   - 61-80: Good match with most requirements met, minor gaps
   - 81-100: Excellent match with all or nearly all requirements met

5. **MISSING CRITICAL SKILLS**:
   - List specific skills/qualifications that are required but not found in the resume
   - Prioritize by importance to the role

6. **HIGHLIGHT OPPORTUNITIES**:
   - Identify transferable skills that could be emphasized
   - Suggest how to frame existing experience to better match the role
   - Note any adjacent skills that could be relevant

7. **RECOMMENDATION**:
   - Should the candidate apply? (Yes/No/Maybe with explanation)
   - What would be the biggest challenges in this role?
   - What would be the candidate's strongest selling points?

FORMAT YOUR RESPONSE AS:
**SKILL MATCH ANALYSIS:**
[List each requirement and match status]

**EXPERIENCE RELEVANCE:**
[Detailed assessment]

**EDUCATION & CERTIFICATIONS:**
[Assessment]

**CONFIDENCE RATING: X/100**
[Explanation of rating]

**MISSING CRITICAL SKILLS:**
[List with priority]

**HIGHLIGHT OPPORTUNITIES:**
[Specific suggestions]

**RECOMMENDATION:**
[Clear yes/no/maybe with reasoning]

Be honest and objective. If the candidate is clearly not qualified, say so clearly and explain why. Don't inflate ratings to be encouraging - accuracy is more valuable than optimism.
"""
            answer = qa_chain.run(prompt)
            st.subheader("✅ AI Analysis")
            st.write(answer)
