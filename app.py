import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit
st.set_page_config(page_title="Vega 0.03", layout="wide")
st.title("🚀 Vega 0.03")

# Load resume from CSV instead of PDF
resume_file = "resume_v1.csv"
if not os.path.exists(resume_file):
    st.error(f"❌ Please make sure {resume_file} exists in this folder!")
    st.stop()

# Load the CSV
df = pd.read_csv(resume_file)

# Convert CSV to a string blob for embedding
resume_text = ""
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

CANDIDATE'S RESUME:
{resume_text}

JOB POSTING TO ANALYZE:
{job_text}

SKILL EQUIVALENCIES - READ CAREFULLY:
When evaluating skills, consider these equivalencies:
- LangChain experience = OpenAI function calling experience
- ChatGPT experience = GPT-4 experience  
- Any database experience (MongoDB, Redis, etc.) = PostgreSQL experience
- Any web framework experience (React, FastAPI, etc.) = FastAPI experience
- Data validation experience = Schema validation experience
- Vector database experience (QDrant, FAISS) = Embeddings experience
- Machine learning experience = Data processing and analysis skills
- AWS services experience = Cloud infrastructure and DevOps skills
- API development experience = Web framework experience
- Backend development experience = FastAPI/ASGI framework experience

IMPORTANT: BE LESS CONSERVATIVE
- If the candidate has mentioned a specific technology in their resume, they have used it at a senior level
- Do not mark skills as "Missing" if the candidate has equivalent or related experience
- When in doubt, mark skills as "Present" rather than "Partially present"
- The candidate's extensive experience suggests they can quickly learn any missing specific tools

CRITICAL: THESE ARE THE SAME SKILLS
The candidate has these technologies in their resume, which ARE the skills the job is asking for:
- LangChain = OpenAI tool-calling paradigm (SAME THING)
- ChatGPT = OpenAI structured response and tool-calling APIs (SAME THING)
- QDrant/FAISS = RAG and embeddings (SAME THING)
- FastAPI = FastAPI/ASGI framework (SAME THING)
- MongoDB/Redis = PostgreSQL + Redis (SAME THING)

DO NOT MARK THESE AS MISSING OR PARTIALLY PRESENT:
- OpenAI tool-calling paradigm (candidate has LangChain)
- OpenAI APIs (candidate has ChatGPT)
- RAG and embeddings (candidate has vector databases)
- FastAPI/ASGI (candidate has FastAPI)
- PostgreSQL (candidate has database experience)

SPECIFIC CANDIDATE SKILL MAPPINGS:
Based on the candidate's resume, these specific experiences count as:
- LangChain work = OpenAI tool-calling paradigm experience
- ChatGPT integration = OpenAI structured response and tool-calling APIs
- QDrant/FAISS vector databases = RAG and embeddings experience
- Knowledge graph development = Agent frameworks experience
- FastAPI projects = FastAPI/ASGI framework experience
- MongoDB/Redis experience = PostgreSQL experience
- Data validation work = Schema-based validation experience

SKILL PRESENCE CRITERIA:
Mark a skill as "Present" if:
- It's explicitly mentioned in the resume
- The candidate has equivalent experience (per equivalencies above)
- The candidate has built systems that clearly require this skill
- The candidate has worked with related technologies that imply this skill

Mark a skill as "Partially present" only if:
- The candidate has some related experience but not the specific skill
- The skill is mentioned but not extensively demonstrated

CONTEXT AWARENESS:
Consider that someone with the candidate's background likely has:
- Database knowledge if they've built data pipelines
- Web framework experience if they've worked with APIs or backend development
- Data processing skills if they've done ML/AI work
- DevOps knowledge if they've built production systems
- ETL skills if they've worked with data engineering
- Backend development skills if they've built data platforms or APIs

PROJECT SKILL EXTRACTION:
From the candidate's projects, extract these implied skills:
- Data engineering projects → Database, ETL, data modeling skills
- AI/ML projects → Python, data processing, model development skills  
- Web development projects → API, framework, frontend skills
- Cloud projects → DevOps, infrastructure, scalability skills
- NLP projects → Text processing, language model skills
- Backend/API projects → Web framework, database, server skills

ANALYSIS INSTRUCTIONS:

1. **SKILL MATCH ANALYSIS** (Be reasonable but accurate):
   - Extract key required skills/qualifications from the job posting
   - Compare each requirement against the resume content
   - For each skill: state if it's present, partially present, or missing
   - Consider related experience and transferable skills
   - Look for equivalent technologies and methodologies
   - Be reasonable about skill variations (e.g., experience with ChatGPT/LangChain counts for GPT-4 experience)
   - Consider implied skills based on project types and technologies used

2. **EXPERIENCE RELEVANCE**:
   - Evaluate if the candidate's work experience directly relates to the job requirements
   - Consider industry relevance, role similarity, and project scope
   - Recognize when experience in similar domains is transferable
   - Distinguish between "directly relevant" and "related" experience

3. **EDUCATION & CERTIFICATIONS**:
   - Check if required degrees/certifications are present
   - Evaluate if the field of study is relevant to the job requirements
   - Consider equivalent qualifications and self-taught expertise

4. **CONFIDENCE RATING CRITERIA** (0-100 scale):
   - 0-20: No relevant skills or experience, completely different field
   - 21-40: Some tangential skills but major gaps in core requirements
   - 41-60: Good foundation, missing some specific tools but core skills present
   - 61-80: Strong match, minor gaps in specific technologies
   - 81-100: Excellent match with all or nearly all requirements met

5. **MISSING CRITICAL SKILLS**:
   - List specific skills/qualifications that are required but not found in the resume
   - Prioritize by importance to the role
   - Distinguish between "missing" and "could be learned quickly"
   - Consider if skills are implied by other experience

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
[List each requirement and match status - consider skill equivalencies and implied skills]

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

GUIDELINES:
- Be reasonable about skill equivalencies (e.g., LangChain experience = LLM experience)
- Consider the candidate's demonstrated ability to learn and adapt
- Recognize when experience in similar domains is highly transferable
- Be honest but fair in your assessment
- Consider implied skills based on project context and technologies used
"""
            answer = llm.predict(prompt)
            st.subheader("✅ AI Analysis")
            st.write(answer)
