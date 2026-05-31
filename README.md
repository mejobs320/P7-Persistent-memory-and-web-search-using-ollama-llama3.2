# Persistent-memory-and-web-search-using-ollama-llama3.2
run ollama locally
program - get restaurant review file xls loaded using vector, than create persistent memory using sqllite and store all prommts and response in that
Logic- search restaurant review from loaded file, from sqllite check prompts, then search restaurant review from google and provide reponse on webui using streamlit

Main py - used tavily web search
Main.py - used duckduckgo_search web search


Prompt:
Create program RAG pipeline created to load restaurant review data using chroma db (code shared in vector.py)
Create persistent memory to store prompts and responses
logic is that we will ask for restaurant review queries which will be first looked into RAG pipeline where doc is provided
Then looked into google search 
Then provide response
Webui is created using streamlit

Demo code to load file is shared
Running locally using ollama (model = OllamaLLM(model="llama3.2"))
Install required libraries
model mxbai-embed-large and llama3.2 is already loaded with ollama installed
