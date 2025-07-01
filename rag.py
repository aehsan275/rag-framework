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

    wiki = wikipediaapi.Wikipedia(user_agent="MyPythonScript/1.0",language="en")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    client = chromadb.PersistentClient(path = "./database")
    collection = client.get_collection("embeddings")
    logger = logging.getLogger("Logger")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",filename="logs.log",filemode="a")

    def get_response(user_input):
        if os.path.getsize("logs.log") > 1000000:
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
                    """
        headers = {"Content-Type":"application_json"}
        try:
            rag.logger.info(f"Input JSON: {json_input}")
            rag.logger.info(f"Output JSON: {requests.post(rag.generative_endpoint, data=json_input, headers=headers).json()}")
            response = requests.post(rag.generative_endpoint, data=json_input, headers=headers).json()["candidates"][0]["content"]["parts"][0]["text"].rstrip()
        except Exception as e:
            rag.logger.info(f"There was an error: {e}")
            response = None

        return response

    def get_embedding(text):
        json_input = """
                    {"model": "models/""" + rag.embedding_model + """",
                    "content": {
                    "parts":[{
                    "text": """ + '"' + text + '"' + """}]}
                    }    
                    """
        headers = {"Content-Type":"application_json"}
        embedding = requests.post(rag.embedding_endpoint, data = json_input, headers=headers).json()["embedding"]["values"]
        return embedding

    def get_context(user_input):
        context = ""
        prompt = f"Your task is to provide the exact titles of up to five of the most relevant Wikipedia pages based on the topic provided below. The titles should be unique, correspond to valid Wikipedia pages, and be separated by a newline. Replace any whitespaces in the titles with underscores. If you can't find five relevant articles, return only the ones you find to be most pertinent. If no articles are highly relevant, just respond with a newline. Here is the topic: <{user_input}> Your response should not contain angle brackets around it. For example, if the title of a wikipedia page is Cars, your response should be Cars, not <Cars>"
        titles = rag.get_response(prompt).split("\n")
        for title in titles:
            content = rag.wiki.page(title).text
            context = context + "\n" + content
        context = context.replace('"',"'")
        return context

    def get_embeddings(user_input, context):
        chunks = rag.text_splitter.split_text(context)
        for i in range(len(chunks)%8):
            chunks.pop(-1)
        chunks = numpy.array(chunks)
        chunks = chunks.reshape(-1,8)
        for pieces in chunks:
            for chunk in pieces:
                results = rag.collection.get(ids=[chunk])
                if len(results["metadatas"]) < 1:
                    embedding = rag.get_embedding(chunk)
                    rag.collection.add(embeddings=[embedding], metadatas=[{"text":chunk}], ids=[chunk])
    
        results = rag.collection.query(query_embeddings=rag.get_embedding(user_input), n_results=5)
        return list((section["text"] for section_list in results["metadatas"] for section in section_list))

    def rag_response(user_input):
        context = "\n".join(rag.get_embeddings(user_input, rag.get_context(user_input)))
        prompt = "Here is some context to help you answer this question. Use information and ideas from this context in your final answer. The context will be delimited by square brackets, and the question will be delimited by angle brackets. Your answer should not be delimited. If the question seems nonsensical, ignore the context and do not mention it in your response. Furthermore, do not mention any bracketed text nor provided text in your response. Do not make any mention at all to the context provided, and ignore it if the user input is nonsensical. Ignore the context if the user input does not make sense. If the user input does not make sense do not mention the context in your response, and pretend it never existed."
        response = rag.get_response(f"{prompt} \n <{user_input}> \n [{context}]")
        return response
    