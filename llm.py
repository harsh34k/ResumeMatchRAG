from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt_template = PromptTemplate(
    input_variables=["matches"],
    template="""
You are ResumeMatchBot, an AI assistant helping HR teams shortlist candidates.

Given these retrieved resumes and metadata:
{matches}

Return a structured JSON with top candidates sorted by fit score.
For each candidate include:
- filename
- summary of skills matched
- missing skills
- overall fit score (0–100)
- short reasoning

Do not hallucinate new information — only use what’s in resumes.
""",
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def analyze_match(matches):
    formatted_matches = "\n".join(
        [f"Resume: {m['metadata']['filename']} | Score: {m['score']}" for m in matches]
    )
    result = chain.invoke({"matches": formatted_matches})
    return result["text"]
