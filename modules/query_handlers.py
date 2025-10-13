

from logger import logger

def query_chain(chain,user_input:str):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        result=chain({"query":user_input})
        # logger.debug("chain executed successfully:{result}")
        print('result',result)

        response={
            "response":result["result"],
            "sources":[doc.metadata.get("file_name","Unknown") for doc in result["source_documents"]]
        }
        logger.debug(f"Chain response:{response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise