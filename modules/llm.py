from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate(
        input_variables=["context", "job_description", "question", "chat_history"],
        template="""
You are **HRBot**, an AI-powered assistant for evaluating resumes against job descriptions.

---

🔍 **Job Description**:
{job_description}

🗂 **Context** (resumes):
{context}

🗣 **Conversation History**:
{chat_history}

🙋‍♂️ **User Question**:
{question}

---

Answer professionally, based only on the context and job description. Always match candidates to the provided job description. For recommendations, rank the top 5 candidates by file_name, providing reasons based on their skills and experience relative to the job description. If insufficient context, say: "I couldn't find enough information to answer."
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, retriever

def get_contextualizer_chain():
    print("[llm.py] Initializing contextualizer chain")
    contextualizer_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    contextualizer_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
You are **ContextualizerBot**, an AI assistant that rewrites follow-up questions into standalone questions.

📜 **Chat History**:
{chat_history}

💬 **New User Question**:
{question}

🧩 **Your Task**:
Rewrite the user's new question so that it makes sense without depending on previous messages.
If it already makes sense, return it as is.
Return **only the rewritten question**.
"""
    )

    return LLMChain(llm=contextualizer_llm, prompt=contextualizer_prompt)
