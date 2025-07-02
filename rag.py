import wikipediaapi
import requests
import chromadb
import numpy
import logging
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

class rag:
    load_dotenv() 

    api_key = os.getenv("API_KEY")
    generative_model = "gemini-1.5-flash"
    generative_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + generative_model + ":generateContent?key=" + api_key
    embedding_model = "embedding-001"
    embedding_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + embedding_model + ":embedContent?key=" + api_key

    wiki = wikipediaapi.Wikipedia(user_agent="MyPythonScript/1.0",language="en") # sets up wikipedia api
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20) # used for chunking
    client = chromadb.PersistentClient(path = "./database") # initializes vector database
    collection = client.get_collection("embeddings")
    logger = logging.getLogger("Logger")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",filename="logs.log",filemode="a")

    def get_response(user_input): # returns the generative model's response to an input
        if os.path.getsize("logs.log") > 1000000: # clears logs if size exceeds 1 mb
            with open("logs.log", "w") as file:
                pass
        json_input = """
                    {
                    "contents": [{
                        "parts": [{
                        "text": """ + '"' + user_input + '"' + """
                        }]
                    }]
                    }
                    """ # json structure that generative model accepts
        headers = {"Content-Type":"application_json"}
        try:
            rag.logger.info(f"Input JSON: {json_input}")
            response = requests.post(rag.generative_endpoint, data=json_input, headers=headers).json()
            rag.logger.info(f"Output JSON: {response}") # logs the input and output
            parsed_response = response["candidates"][0]["content"]["parts"][0]["text"].rstrip() # parses json response
        except Exception as e:
            rag.logger.info(f"There was an error: {e}")
            parsed_response = None # if there is an error there is no response

        return parsed_response

    def get_embedding(text): # returns the embedding model's response to an input
        json_input = """
                    {"model": "models/""" + rag.embedding_model + """",
                    "content": {
                    "parts":[{
                    "text": """ + '"' + text + '"' + """}]}
                    }    
                    """ # json structure that embedding model accepts
        headers = {"Content-Type":"application_json"}
        embedding = requests.post(rag.embedding_endpoint, data = json_input, headers=headers).json()["embedding"]["values"] #parses json response
        return embedding

    def get_context(user_input): # gets relevant context from wikipedia based on an input
        context = ""
        prompt = f"Your task is to provide the exact titles of up to five of the most relevant Wikipedia pages based on the topic provided below. The titles should be unique, correspond to valid Wikipedia pages, and be separated by a newline. Replace any whitespaces in the titles with underscores. If you can't find five relevant articles, return only the ones you find to be most pertinent. If no articles are highly relevant, just respond with a newline. Here is the topic: <{user_input}> Your response should not contain angle brackets around it. For example, if the title of a wikipedia page is Cars, your response should be Cars, not <Cars>" # using prompt engineering to get reliable output from LLM
        titles = rag.get_response(prompt).split("\n") # converts LLM response into a list
        for title in titles:
            content = rag.wiki.page(title).text
            context = context + "\n" + content
        context = context.replace('"',"'") # replaces double quotes with single quotes so json format is not broken when context is passed to LLM
        return context

    def get_embeddings(user_input, context): # finds relevant chunks based on context and an input
        chunks = rag.text_splitter.split_text(context)
        for i in range(len(chunks)%8):
            chunks.pop(-1)
        chunks = numpy.array(chunks)
        chunks = chunks.reshape(-1,8) # this bypasses the embedding model's limits for 8 embeddings per request by creating multiple requests
        for pieces in chunks:
            for chunk in pieces:
                results = rag.collection.get(ids=[chunk])
                if len(results["metadatas"]) < 1: # checks if an embedding has already been made
                    embedding = rag.get_embedding(chunk)
                    rag.collection.add(embeddings=[embedding], metadatas=[{"text":chunk}], ids=[chunk])
    
        results = rag.collection.query(query_embeddings=rag.get_embedding(user_input), n_results=5)
        return list((section["text"] for section_list in results["metadatas"] for section in section_list)) # returns 5 most relevant chunks

    def rag_response(user_input): # puts the whole thing together
        context = "\n".join(rag.get_embeddings(user_input, rag.get_context(user_input)))
        prompt = "Here is some context to help you answer this question. Use information and ideas from this context in your final answer. The context will be delimited by square brackets, and the question will be delimited by angle brackets. Your answer should not be delimited. If the question seems nonsensical, ignore the context and do not mention it in your response. Furthermore, do not mention any bracketed text nor provided text in your response. Do not make any mention at all to the context provided, and ignore it if the user input is nonsensical. Ignore the context if the user input does not make sense. If the user input does not make sense do not mention the context in your response, and pretend it never existed."
        response = rag.get_response(f"{prompt} \n <{user_input}> \n [{context}]")
        return response
    