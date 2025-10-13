from logger import logger
from langchain_core.documents import Document

def query_chain(chain_tuple, user_input: str, jd_text: str):
    try:
        chain, retriever = chain_tuple  # Unpack chain and retriever
        logger.debug(f"Running chain for input: {user_input}, JD: {jd_text[:100]}...")
        print(f"[query_handlers.py] Chain input_variables expected: {chain.prompt.input_variables}")  # Debug
        
        # Validate user input
        if not user_input or not isinstance(user_input, str):
            raise ValueError(f"Invalid user_input: {user_input!r}")
        
        # If job_description missing, handle gracefully
        if not jd_text or not isinstance(jd_text, str) or jd_text.strip() == "":
            print("[query_handlers.py] ⚠️ No job description found, continuing with empty context.")
            jd_text = "No job description was provided for this query."

        # Retrieve matching resume content
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        print(f"[query_handlers.py] Retrieved context of length {len(context)}")  # Debug
        
        # Build input for LLM
        input_dict = {"question": user_input, "job_description": jd_text, "context": context}
        print(f"[query_handlers.py] Invoking chain with input_dict")  # Debug
        result = chain.invoke(input_dict)
        
        response = {
            "response": result["text"],
            "sources": [doc.metadata.get("file_name", "Unknown") for doc in docs]
        }
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.exception(f"Error in query_chain: {str(e)}")
        raise
