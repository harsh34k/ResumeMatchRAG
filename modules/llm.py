from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
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
        input_variables=["context", "question"],
        template="""
You are **HRBot**, an AI-powered assistant trained to help HR teams evaluate and query candidate resumes.

Your job is to provide clear, accurate, and helpful responses based **only on the provided context** from the resumes.

The context may include chunks from multiple resumes—group information by 'file_name' when comparing candidates.

---

🔍 **Context** (from resumes):
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer**:
- Respond in a professional, factual, and neutral tone.
- Use simple explanations for skills, experience, or qualifications.
- If comparing candidates (e.g., "which one is good?"), reference each by 'file_name', list pros/cons, and base on context only.
- If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the uploaded resumes."
- Do NOT make up facts or assume details not in the context.
- Do NOT give hiring advice or biases—stick to factual summaries.
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )












