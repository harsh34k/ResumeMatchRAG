from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    print("[llm.py] Initializing LLM chain")  # Debug
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate(
        input_variables=["context", "job_description", "question"],
        template="""
You are **HRBot**, an AI-powered assistant for evaluating resumes against job descriptions.

---

🔍 **Job Description**:
{job_description}

🗂 **Context** (resumes):
{context}

🙋‍♂️ **User Question**:
{question}

---

Answer professionally, based only on the context and job description. Always match candidates to the provided job description. For recommendations, rank the top 5 candidates by file_name, providing reasons based on their skills and experience relative to the job description. If insufficient context, say: "I couldn't find enough information to answer."
"""
    )
    print(f"[llm.py] Prompt created with input_variables: {prompt.input_variables}")  # Debug

    chain = LLMChain(llm=llm, prompt=prompt)
    print(f"[llm.py] LLMChain created with input_variables: {prompt.input_variables}")  # Debug
    return chain, retriever  # Return both chain and retriever