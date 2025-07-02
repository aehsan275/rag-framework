# Retrieval Augmented Generation Framework

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines the power of a large language model (Gemini 1.5) with live content from Wikipedia to answer user questions with grounded, up-to-date information. The system is wrapped in a user-friendly Streamlit interface and includes persistent vector storage via ChromaDB.

#### When a user submits a query, the system:

* Generates relevant Wikipedia article titles using Gemini.
* Fetches article content using the Wikipedia API.
* Splits the content into manageable chunks using LangChain’s RecursiveCharacterTextSplitter.
* Generates vector embeddings for each chunk using Gemini’s embedding API.
* Stores and retrieves embeddings via ChromaDB to avoid recomputation.
* Selects the top 5 most relevant chunks based on vector similarity to the user query.
* Generates a final answer using Gemini’s generative API, informed by the retrieved context.
* The system includes basic logging, chunk deduplication, and embedding caching to optimize performance over repeated queries. Logs are viewable within the Streamlit UI and automatically truncate when too large.

This RAG implementation is ideal for experimentation with generative retrieval pipelines, prototyping LLM-backed QA systems, or understanding how structured context can improve factual accuracy in language model outputs.

## What This System Can Do

* **Answer user questions using Wikipedia-sourced content**

&nbsp;  The system pulls information directly from Wikipedia articles relevant to your query and uses it as context for Gemini's final answer.

* **Generate relevant Wikipedia article titles**

&nbsp;  It prompts Gemini to return the top Wikipedia page titles based on your input topic.

* **Embed and store contextual chunks for retrieval**

&nbsp;  It splits large texts into chunks, embeds them using Gemini, and stores them persistently in ChromaDB for future reuse.

* **Retrieve the most relevant chunks based on vector similarity**

&nbsp;  It selects top-matching chunks from ChromaDB using cosine similarity against the query embedding.

* **Prevent redundant embedding of repeated content**

&nbsp;  It checks ChromaDB before re-embedding content, avoiding unnecessary API calls.

* **Log input/output data for inspection and debugging**

&nbsp;  All API inputs and responses are logged in logs.log and viewable within the Streamlit UI.

## What This System Cannot Do

* **Handle multi-turn conversations or follow-up questions**

&nbsp;  The system is stateless and only processes one query at a time without memory of past interactions.

* **Validate or correct hallucinated Wikipedia titles**

&nbsp;  Gemini may return Wikipedia titles that don’t exist or aren’t highly relevant; these are fetched blindly.

* **Filter or rank results using advanced retriever logic**

&nbsp;  It uses simple cosine similarity without re-ranking, hybrid search, or learned retrieval.

* **Guarantee factual correctness beyond Wikipedia**

&nbsp;  It only uses Wikipedia as a context source — no other web search, databases, or external knowledge is queried.

* **Support high-concurrency or real-time performance**

&nbsp;  It uses synchronous requests, blocking API calls, and has no background processing or queue system.

## Setting Up

First, clone the repository and install the dependencies.

```bash

git clone https://github.com/aehsan275/rag-framework

cd rag-framework

pip install -r requirements.txt

```

Then, add your API key to the .env file

```env
API_KEY=your_api_key_here
```

Then, to run the project,

```env
streamlit run gui.py
```

A Streamlit window will automatically open in your browser.

## Usage

To use the application, simply navigate to the "**Model**" tab and enter your prompt in the text box. Click Submit and wait for your answer.

To view the logs, navigate to the "**Logs"** tab to see the logs. The logs can be cleared by clicking the Clear Logs button.
