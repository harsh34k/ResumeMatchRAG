from logger import logger
from langchain_core.documents import Document

def query_chain(chain_tuple, user_input: str, jd_text: str):
    try:
        chain, retriever = chain_tuple  # Unpack chain and retriever
        logger.debug(f"Running chain for input: {user_input}, JD: {jd_text}")
        print(f"[query_handlers.py] Chain input_variables expected: {chain.prompt.input_variables}")  # Debug
        print(f"[query_handlers.py] Input provided: {{'question': {user_input!r}, 'job_description': {jd_text!r}}}")  # Debug
        
        # Validate inputs
        if not user_input or not isinstance(user_input, str):
            raise ValueError(f"Invalid user_input: {user_input!r}")
        if not jd_text or not isinstance(jd_text, str):
            raise ValueError(f"Invalid jd_text: {jd_text!r}")
        
        # Manually retrieve documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        print(f"[query_handlers.py] Retrieved context: {context}...")  # Debug
        
        # Pass all inputs to LLMChain
        input_dict = {"question": user_input, "job_description": jd_text, "context": context}
        print(f"[query_handlers.py] Invoking chain with: {input_dict}")  # Debug
        result = chain.invoke(input_dict)
        
        print(f"[query_handlers.py] Result: {result}")  # Debug
        response = {
            "response": result["text"],
            "sources": [doc.metadata.get("file_name", "Unknown") for doc in docs]
        }
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.exception(f"Error in query_chain: {str(e)}")
        raise